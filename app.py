import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import os

from flask import Flask, request, jsonify

import google.generativeai as genai
from IPython.display import Markdown
import PIL.Image

openai.api_key = os.getenv("OPENAI_API_KEY")
gem_api_key = os.getenv("GEM_API_KEY")

datafile_path = "rendi_embedding.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)


def search_clothes(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    
    # similarity 계산
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    # similarity에 따라 정렬하고 필요한 열 선택
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)[['title', 'similarity', 'combined']]  # 'combined' 열 포함
        .reset_index()
    )

    # 결과가 비어 있지 않은지 확인
    if results.empty:
        print("No results found.")
        return {
            "titles": [],
            "similarities": []
        }

    titles = []
    similarities = []

    if pprint:
        for i, row in results.iterrows():
            title = row['title']  # 'title' 열에서 제목 가져오기
            similarity = row['similarity']  # 'similarity' 값 가져오기
            titles.append(title)
            similarities.append(similarity)

    # 딕셔너리 형태로 반환
    return {
        "titles": titles,
        "similarities": similarities
    }
    
def generate_fashion_analysis(api_key, image_path):
    # API 키 설정
    genai.configure(api_key=api_key)
    # 이미지 파일 열기
    userImageFile = PIL.Image.open(image_path)
    # 모델 초기화
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # 콘텐츠 생성 요청
    response = model.generate_content([userImageFile, "너는 패션 분석가로서 100자 이내로 분석해주고 뉘앙스적인 것도 포함되었으면 해. 답변은 개조식의 긴 하나의 문장으로 부탁"])
    # 응답의 텍스트 반환
    return response.text



app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    print("flask 시작 제발!")
    
    try:
        data = request.get_json()
        keyword_name = data['name']
        print("1")
        print(openai.api_key)
        print(gem_api_key)
        print('key 출력 완료')
        
        results = search_clothes(df, keyword_name, n=48)
        print("2")
        
        # 반환된 결과 출력
        print("Returned Titles:", results["titles"])
        print("Returned Similarities:", results["similarities"])

        # titles와 similarities를 직접 가져오기
        titles = results["titles"]  # 수정된 부분
        similarities = results["similarities"]  # 수정된 부분

        return jsonify({'titles': titles, 'similarities': similarities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/imageSearch', methods=['POST'])
def imageSearch():
    print("Flask 이미지 검색 시작!")
    
    # 이미지 파일이 요청에 포함되어 있는지 확인
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    print('이미지 파일 확인 완료.')
    imageFile = request.files['image']

    # 이미지 파일이 비어있지 않은지 확인
    if imageFile.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    # 이미지 분석 요청
    try:
        response_text = generate_fashion_analysis(gem_api_key, imageFile.stream)
        print("응답 텍스트:", response_text)
        
        # 결과 검색
        results = search_clothes(df, response_text, n=48)
        print('키워드 검색 진행 중...')
        
        # 결과 추출
        titles = results.get("titles", [])
        similarities = results.get("similarities", [])
        
        print("반환된 제목들:", titles)
        print("반환된 유사도:", similarities)

        return jsonify({'titles': titles, 'similarities': similarities})
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return jsonify({"error": "An error occurred during processing."}), 500
    
    
@app.route('/health')
def health_check():
    return 'flask 성공'

@app.route('/check_env')
def check_env():
    return f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}, GEM_API_KEY: {os.getenv('GEM_API_KEY')}"
