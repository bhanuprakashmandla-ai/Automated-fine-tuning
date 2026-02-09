"""Data preparation and QA generation pipeline"""

import os
import warnings
import tempfile
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, Dataset
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.parsers.pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class QAGenerationPipeline:
    def __init__(self, llm_config):
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing LLM API key. Set 'api_key' in pdf_config.llm_config or "
                "define OPENAI_API_KEY/LLM_API_KEY environment variable."
            )
        self.client = LLMClient(
            api_base=llm_config['api_base'],
            model_name=llm_config['model_name'],
            api_key=api_key
        )
        self.generator = QAGenerator(client=self.client)

    def extract_pdf_text(self, pdf_content: bytes) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "temp.pdf")
            try:
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(pdf_content)
                parser = PDFParser()
                text = parser.parse(temp_file_path)
                os.remove(temp_file_path)
                os.rmdir(temp_dir)
                return text
            except Exception as e:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                raise e

    def chunk_text(self, text: str, max_sequence_lenght: int = 2048, overlap: int = 200) -> List[str]:
        return split_into_chunks(text, chunk_size=max_sequence_lenght, overlap=overlap)

    def generate_qa_pairs_from_text(self, chunks: List[str], num_pairs: int = 5, max_generation_tokens: int = 512, max_workers: int = 5) -> List[dict]:
        if hasattr(self.client, "config"):
            self.client.config.setdefault("generation", {})
            self.client.config["generation"]["max_tokens"] = max_generation_tokens

        def process_chunk(i, chunk):
            try:
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except:
                    pass
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                summary = self.generator.generate_summary(document_text=chunk)
                qa_pairs = self.generator.generate_qa_pairs(num_pairs=num_pairs, document_text=chunk, summary=summary)
                return qa_pairs, None
            except Exception as e:
                return None, f"Error processing chunk {i+1}: {str(e)}"

        all_qa_pairs = []
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                qa_pairs, error = future.result()
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                if error:
                    logger.error(error)
                    errors.append(error)

        if not all_qa_pairs and errors:
            raise Exception(f"Failed to generate any QA pairs. Errors: {'; '.join(errors)}")
        return all_qa_pairs


class DataPreparation:
    def _split_dataset_to_trian_test(self, dataset):
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        return split_dataset['train'], split_dataset['test']

    def _combine_fields(self, dataset, input_fields, output_fields):
        available_columns = dataset.column_names
        for field in input_fields + output_fields:
            if field not in available_columns:
                raise ValueError(f"Field '{field}' not found in dataset columns: {available_columns}")

        def combine_examples(example):
            question = " ".join(str(example[f]).strip() for f in input_fields if example.get(f))
            answer = " ".join(str(example[f]).strip() for f in output_fields if example.get(f))
            return {"question": question, "answer": answer}

        dataset = dataset.map(combine_examples, num_proc=os.cpu_count(), load_from_cache_file=True, desc="Combining input/output fields")
        return dataset.select_columns(["question", "answer"])

    def _generate_qa_pairs_from_pdf(self, path, type, pdf_config):
        llm_config = pdf_config['llm_config']
        qa_generator = QAGenerationPipeline(llm_config)
        
        def generate_pairs(pdf_path):
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
            with open(pdf_path, "rb") as file:
                pdf_content = file.read()
            text = qa_generator.extract_pdf_text(pdf_content)
            logger.info(f"Extracted text length: {len(text)}")
            chunks = qa_generator.chunk_text(text=text, max_sequence_lenght=pdf_config['chunk_size'], overlap=pdf_config['overlap'])
            logger.info(f"Number of chunks: {len(chunks)}")
            qa_pairs = qa_generator.generate_qa_pairs_from_text(chunks, pdf_config['qa_pairs_per_chunk'], pdf_config['max_generation_tokens'], max_workers=10)
            logger.info(f"Generated {len(qa_pairs)} QA pairs from {os.path.basename(pdf_path)}")
            return qa_pairs

        if type == "file":
            return generate_pairs(path)
        elif type == "folder":
            if not os.path.isdir(path):
                raise ValueError(f"Provided path is not a folder: {path}")
            all_qa_pairs = []
            for filename in os.listdir(path):
                if filename.lower().endswith(".pdf"):
                    pdf_file = os.path.join(path, filename)
                    logger.info(f"Processing PDF: {filename}")
                    try:
                        qa_pairs = generate_pairs(pdf_file)
                        for qa in qa_pairs:
                            qa["source_file"] = filename
                        all_qa_pairs.extend(qa_pairs)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
            logger.info(f"Total QA pairs generated from folder: {len(all_qa_pairs)}")
            return all_qa_pairs
        else:
            raise ValueError(f"Invalid type: {type}. Must be 'file' or 'folder'")

    def get_train_test_data(self, dataset_config):
        if dataset_config['type'] == "huggingface":
            hf_token = (
                dataset_config.get('hf_token')
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_TOKEN")
            )
            dataset = load_dataset(dataset_config['path'], split="train", token=hf_token)
        elif dataset_config['splitter'] == "pdf":
            qa_pairs = self._generate_qa_pairs_from_pdf(dataset_config['path'], dataset_config['type'], dataset_config['pdf_config'])
            dataset = Dataset.from_list(qa_pairs)
        elif dataset_config['type'] == "file":
            dataset = load_dataset(dataset_config['splitter'], data_files=dataset_config["path"], split="train")

        dataset = self._combine_fields(dataset, dataset_config['input_fields'], dataset_config['output_fields'])
        train_data, test_data = self._split_dataset_to_trian_test(dataset)
        train_data = train_data.shuffle(seed=42)
        test_data = test_data.shuffle(seed=42)

        total_train = len(train_data)
        batch_config = dataset_config['batch_config']
        first_size = int(batch_config['first_batch'] * total_train)
        second_size = int(batch_config['second_batch'] * total_train)
        third_size = int(batch_config['third_batch'] * total_train)
        test_size = int(batch_config['test_batch'] * len(test_data))

        return {
            "first_batch": train_data.select(range(first_size)),
            "second_batch": train_data.select(range(first_size, first_size + second_size)),
            "third_batch": train_data.select(range(first_size + second_size, first_size + second_size + third_size)),
            "test_data": test_data.select(range(test_size))
        }
