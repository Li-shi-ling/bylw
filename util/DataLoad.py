from PyPDF2 import PdfFileReader
from docx import Document
import csv
import os
import re

class DocumentReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.read_document()

    def read_txt(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            return ""

    def read_docx(self):
        try:
            doc = Document(self.file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return ""

    def read_pdf(self):
        try:
            pdf = PdfFileReader(open(self.file_path, 'rb'))
            text = ''
            for page_num in range(pdf.numPages):
                text += pdf.getPage(page_num).extractText()
            return text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""

    def read_csv(self):
        try:
            with open(self.file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = [str(dict(row)) for row in reader]
                return '\n'.join(rows)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return ""

    def read_document(self):
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()
        if ext == '.txt':
            return self.read_txt()
        elif ext == '.docx':
            return self.read_docx()
        elif ext == '.pdf':
            return self.read_pdf()
        elif ext == '.csv':
            return self.read_csv()
        else:
            return ""

    def preprocess_text(self, text):
        try:
            # 分段
            paragraphs = []
            for i in text.split('\n'):
                if i != '' and len(i) > 10:
                    paragraphs.append(i)
            # 分句
            sentences = [re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para) for para in paragraphs]
            # 将分句的结果扁平化
            sentences = [sentence for sublist in sentences for sentence in sublist]
            return sentences
        except Exception as e:
            print(f"Error during text preprocessing: {e}")
            return []

    def get_rag_input(self):
        preprocessed_text = self.preprocess_text(self.text)
        return preprocessed_text

if __name__ == "__main__":
    # file_path = '../data/datafrom/word/mydoc.docx'
    # Embedding = STEmbedding(modelpath="../data/paraphrase-multilingual-MiniLM-L12-v2/")
    # vectordb = VectorDB(Embedding)
    #
    # doc_reader = DocumentReader(file_path)
    # rag_input = doc_reader.get_rag_input()
    #
    # vectordb.store(rag_input,"../data/knowledgeBases/base.csv")
    from model.TextVectorization.SentenceTransformer import STEmbedding
    from model.VectorizedTextSimilarityMatching.VectorDB import VectorDB
    import pandas as pd
    import numpy as np
    file_path = '../data/Gdata/Graph.csv'
    Embedding = STEmbedding(modelpath="../data/paraphrase-multilingual-MiniLM-L12-v2/")
    vectordb = VectorDB(Embedding)
    data = np.array(pd.read_csv(file_path)[["实体a","关系","实体b"]]).tolist()
    vectordb.store(["".join(d) for d in data], "../data/knowledgeBases/base.csv")
