from model.TextVectorization.base.TextVectorizationBase import TextVectorizationBase
import tensorflow_hub as hub
import tensorflow as tf

class USETextVectorizer(TextVectorizationBase):
    def __init__(self):
        # 加载Universal Sentence Encoder模型
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    def embed_documents(self, documents: list):
        # 将一列表数据处理为一列表向量
        embeddings = self.model(documents)
        return embeddings.numpy().tolist()

    def embed_query(self, text: str):
        # 将一段数据处理为一个向量
        embedding = self.model([text])
        return embedding.numpy().tolist()[0]


# 示例使用
if __name__ == "__main__":
    vectorizer = USETextVectorizer()
    documents = ["这是一个测试文档。", "通用句子编码器非常强大。"]
    query = "通用句子编码器有多强大？"

    doc_embeddings = vectorizer.embed_documents(documents)
    query_embedding = vectorizer.embed_query(query)

    print("Document Embeddings:", doc_embeddings)
    print("Query Embedding:", query_embedding)