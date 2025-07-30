import streamlit as st
import os
import io
import fitz  # PyMuPDF
import numpy as np
import faiss
from PIL import Image
import pickle
import time
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
import shutil
import tempfile
from tqdm import tqdm
import base64
import matplotlib.pyplot as plt
import gc

# 设置页面配置
st.set_page_config(
    page_title="PDF图像搜索引擎",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 忽略不必要的警告
import warnings
warnings.filterwarnings("ignore")

# 模型缓存路径
MODEL_CACHE = "./model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)

class DINOV2FeatureExtractor:
    def __init__(self, model_size="small"):
        """
        初始化DINOv2特征提取器
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_name = {
            "small": "/Users/qiangqianghu/Projects/huggingface_models_bin/dinov2_small",
            "base": "/Users/qiangqianghu/Projects/huggingface_models_bin/dinov2_base",
            "large": "facebook/dinov2-large",
            "giant": "facebook/dinov2-giant"
        }[model_size]
        
        st.info(f"加载DINOv2模型: {model_name}...")
        start_time = time.time()
        
        # 使用本地缓存
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        # macOS 上使用 MPS 加速（Apple Silicon）
        if torch.backends.mps.is_available():
            print("使用 Apple MPS 加速")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:
            print("使用 CPU")
            self.model = AutoModel.from_pretrained(model_name)
        # 设置模型为评估模式   
        self.model.eval()
        
        # 创建图像预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # DINOv2的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        st.success(f"模型加载完成! 耗时: {time.time()-start_time:.2f}秒")

    def extract_features(self, img):
        """使用DINOv2提取图像特征"""
        try:
            # 预处理图像
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(img)
            
            # 获取[CLS]标记的特征
            features = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # 归一化特征向量
            features = features / np.linalg.norm(features)
            
            return features.astype('float32').flatten()
        
        except Exception as e:
            st.error(f"特征提取错误: {str(e)}")
            return np.zeros(self.model.config.hidden_size, dtype='float32')

class PDFImageSearcher:
    def __init__(self, index_path="dino_pdf_index.index", metadata_path="dino_metadata.pkl", model_size="small"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.feature_extractor = DINOV2FeatureExtractor(model_size)
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        self.model_size = model_size
        self.index_loaded = False
    
    def build_index(self, pdf_directory):
        """构建PDF图像索引"""
        st.info(f"开始构建PDF图像索引，目录: {pdf_directory}")
        st.info(f"使用特征维度: {self.feature_dim}")
        
        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.warning("未找到PDF文件")
            return
        
        # 准备FAISS索引 - 使用内积计算余弦相似度
        self.index = faiss.IndexFlatIP(self.feature_dim)
        self.metadata = []
        
        total_images = 0
        start_time = time.time()
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 遍历所有PDF文件
        for idx, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                status_text.text(f"处理文件中: {pdf_file} ({idx+1}/{len(pdf_files)})")
                doc = fitz.open(pdf_path)
                # 遍历所有页面
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    img_list = page.get_images(full=True)
                    
                    # 处理页面中的每张图片
                    for img_index, img_info in enumerate(img_list):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img["image"]
                        
                        # 转换为PIL图像
                        try:
                            img = Image.open(io.BytesIO(img_bytes))
                            # 转换为RGB格式
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                                
                            # 调整大图像尺寸（避免内存问题）
                            if max(img.size) > 2000:
                                ratio = 2000 / max(img.size)
                                new_size = (int(img.width * ratio), int(img.height * ratio))
                                img = img.resize(new_size, Image.LANCZOS)
                                
                            # 使用DINOv2提取特征
                            features = self.feature_extractor.extract_features(img)
                            
                            # 添加到索引
                            self.index.add(np.array([features]))
                            
                            # 存储元数据
                            self.metadata.append({
                                "book": pdf_file,
                                "page": page_num + 1,
                                "img_index": img_index,
                                "dimensions": img.size,
                                "image_bytes": img_bytes  # 存储原始图像字节
                            })
                            
                            total_images += 1
                        except Exception as e:
                            st.warning(f"处理图像时出错 {pdf_file} 第 {page_num+1} 页: {str(e)}")
                
                doc.close()
                progress_bar.progress((idx + 1) / len(pdf_files))
                # 定期清理内存
                if total_images % 10 == 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
            except Exception as e:
                st.error(f"打开PDF时出错 {pdf_file}: {str(e)}")
        
        # 保存索引和元数据
        if total_images > 0:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            st.success(f"索引构建完成! 共处理 {len(pdf_files)} 本PDF, {total_images} 张图片")
            st.success(f"耗时: {time.time()-start_time:.2f}秒")
            self.index_loaded = True
        else:
            st.warning("未找到可处理的图像")
            self.index = None
            
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_index(self):
        """加载预构建的索引"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            st.info("加载索引文件...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            st.success(f"索引加载成功! 共 {len(self.metadata)} 张图片")
            self.index_loaded = True
            return True
        else:
            st.warning("未找到索引文件，请先构建索引")
            return False

    def search_image(self, image, top_k=5, similarity_threshold=70.0):
        """搜索相似图片"""
        if not self.index_loaded:
            if not self.load_index():
                return []
        
        try:
            # 转换图像
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 提取特征
            target_features = self.feature_extractor.extract_features(image)
            
            # 在索引中搜索
            similarities, indices = self.index.search(
                np.array([target_features]).astype('float32'), 
                top_k
            )
            
            # 准备结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                match = self.metadata[idx]
                # 余弦相似度 = 内积（因为特征向量已归一化）
                cosine_similarity = float(similarities[0][i])
                
                # 转换为百分比相似度 (0-100%)
                similarity_percent = max(0.0, min(100.0, cosine_similarity * 100))
                
                if similarity_percent >= similarity_threshold:
                    # 从字节创建图像
                    match_image = Image.open(io.BytesIO(match["image_bytes"]))
                    
                    results.append({
                        "book": match["book"],
                        "page": match["page"],
                        "similarity": similarity_percent,
                        "dimensions": match["dimensions"],
                        "image": match_image
                    })
            
            # 按相似度排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results
        except Exception as e:
            st.error(f"搜索时出错: {str(e)}")
            return []

# 应用主函数
def main():
    # 标题
    st.title("📚 PDF图像搜索引擎")
    st.markdown("使用DINOv2深度学习模型在PDF库中搜索相似图像")
    
    # 初始化状态
    if 'searcher' not in st.session_state:
        st.session_state.searcher = PDFImageSearcher()
    
    # 侧边栏 - 索引管理
    with st.sidebar:
        st.header("索引管理")
        
        # 模型选择
        model_size = st.selectbox(
            "选择DINOv2模型尺寸",
            ("small", "base", "large", "giant"),
            index=0,
            help="small: 速度快但精度较低, giant: 精度高但速度慢"
        )
        
        # 更新模型
        if st.button("重新加载模型"):
            st.session_state.searcher = PDFImageSearcher(model_size=model_size)
            st.success("模型已重新加载!")
        
        # 文件上传
        st.subheader("创建新索引")
        uploaded_files = st.file_uploader(
            "上传PDF文件", 
            type="pdf", 
            accept_multiple_files=True,
            help="选择要添加到索引的PDF文件"
        )
        
        # 构建索引
        if uploaded_files and st.button("构建索引"):
            with st.spinner("正在处理PDF文件..."):
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                
                # 保存上传的文件
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # 构建索引
                st.session_state.searcher = PDFImageSearcher(model_size=model_size)
                st.session_state.searcher.build_index(temp_dir)
                
                # 清理临时文件
                shutil.rmtree(temp_dir)
        
        # 索引状态
        st.subheader("索引信息")
        if st.session_state.searcher.index_loaded:
            st.success("索引已加载")
            st.info(f"图片数量: {len(st.session_state.searcher.metadata)}")
            st.info(f"模型尺寸: {st.session_state.searcher.model_size}")
        else:
            st.warning("未加载索引")
        
        # 示例图片
        st.subheader("使用示例")
        st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=300", 
                 caption="示例图片", use_container_width=True)
        st.markdown("上传类似图片进行搜索")
        
        # 重置按钮
        if st.button("重置应用"):
            st.session_state.clear()
            st.experimental_rerun()
    
    # 主内容区 - 图片搜索
    st.header("图像搜索")
    
    # 图片上传
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_image = st.file_uploader(
            "上传搜索图片", 
            type=["jpg", "jpeg", "png"],
            help="上传要在PDF库中搜索的图片"
        )
        
        # 搜索参数
        top_k = st.slider("返回结果数量", 1, 20, 5)
        similarity_threshold = st.slider("相似度阈值 (%)", 0, 100, 70)
        
        # 搜索按钮
        if st.button("开始搜索", disabled=not st.session_state.searcher.index_loaded):
            if uploaded_image:
                with st.spinner("正在搜索..."):
                    try:
                        image = Image.open(uploaded_image)
                        st.session_state.results = st.session_state.searcher.search_image(
                            image, 
                            top_k=top_k,
                            similarity_threshold=similarity_threshold
                        )
                    except Exception as e:
                        st.error(f"图片处理错误: {str(e)}")
            else:
                st.warning("请先上传图片")
    
    with col2:
        if uploaded_image:
            st.image(uploaded_image, caption="上传的图片", use_container_width=True)
        else:
            st.info("请上传搜索图片")
    
    # 显示结果
    if 'results' in st.session_state and st.session_state.results:
        st.header("搜索结果")
        st.success(f"找到 {len(st.session_state.results)} 个匹配结果")
        
        # 显示结果网格
        cols = st.columns(min(3, len(st.session_state.results)))
        
        for idx, result in enumerate(st.session_state.results):
            col = cols[idx % len(cols)]
            
            with col:
                st.image(
                    result["image"], 
                    caption=f"相似度: {result['similarity']:.2f}%",
                    use_container_width=True
                )
                
                st.markdown(f"**电子书**: `{result['book']}`")
                st.markdown(f"**页码**: {result['page']}")
                st.markdown(f"**原始尺寸**: {result['dimensions'][0]}×{result['dimensions'][1]}")
                
                # 添加下载按钮
                buffered = io.BytesIO()
                result["image"].save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                href = f'<a href="data:image/jpeg;base64,{img_str}" download="match_{idx+1}.jpg">下载图片</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.divider()
    elif 'results' in st.session_state:
        st.warning("未找到匹配结果")

# 运行应用
if __name__ == "__main__":
    main()