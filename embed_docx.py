import os
os.environ["USE_TORCH"] = "1"  # PyTorch 강제 사용

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm

# 🔍 .docx 파일 가져오기
def get_docx_files():
    current_dir = os.getcwd()
    docx_files = [f for f in os.listdir(current_dir) if f.endswith(".docx")]
    return docx_files

# 🧩 구조 기반 청킹
def custom_chunk_by_structure(text: str):
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    section_title = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("▶"):
            if current_chunk:
                chunks.append((section_title, current_chunk.strip()))
                current_chunk = ""
            section_title = stripped
            current_chunk += stripped + "\n"
        elif stripped.startswith("-") or stripped.startswith("■"):
            current_chunk += stripped + "\n"
        else:
            current_chunk += stripped + "\n"

    if current_chunk:
        chunks.append((section_title, current_chunk.strip()))
    return chunks

# 📄 .docx 로드 및 청킹
def load_and_split_docx(docx_files):
    documents = []
    for docx_file in tqdm(docx_files, desc="Loading and Chunking DOCX files"):
        try:
            loader = UnstructuredWordDocumentLoader(docx_file)
            raw_docs = loader.load()
            for doc in raw_docs:
                chunks = custom_chunk_by_structure(doc.page_content)
                for section, chunk in chunks:
                    metadata = {"source": docx_file, "section": section}
                    documents.append(Document(page_content=chunk, metadata=metadata))
        except Exception as e:
            print(f"Error processing {docx_file}: {e}")
    return documents

# 📁 저장 폴더 확인
def ensure_output_folder(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output folder: {output_path}")
    else:
        print(f"Output folder already exists: {output_path}")

# ✅ 메인 함수
def main():
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    docx_files = get_docx_files()
    if not docx_files:
        print("No DOCX files found in the current directory.")
        return
    print(f"Found {len(docx_files)} DOCX files: {docx_files}")
    documents = load_and_split_docx(docx_files)
    if not documents:
        print("No documents were successfully loaded.")
        return
    print(f"Total {len(documents)} document chunks created.")
    print("Loading embedding model: jhgan/ko-sroberta-multitask...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
    print("Creating FAISS index...")
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return
    output_path = os.path.join(current_dir, "faiss_index")
    ensure_output_folder(output_path)
    try:
        vectorstore.save_local(output_path)
        print(f"✅ FAISS index saved to: {output_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        print("Please check write permissions, disk space, or FAISS installation.")

if __name__ == "__main__":
    main()
