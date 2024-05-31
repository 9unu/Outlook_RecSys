from bs4 import BeautifulSoup
import requests
import time
import random
import time
import warnings
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import re
import json
from fake_useragent import UserAgent


# r = requests.get(url, headers=headers)
def musinsa_Crawling():
    # 헤더 설정
    headers = {'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    url=f"https://www.musinsa.com/app/styles/lists?&sort=NEWEST&page=1"
    response=requests.get(url, headers=headers)
    html=response.text
    soup = BeautifulSoup(html, "html.parser")
    total_page_num = int(soup.find("span", class_="totalPagingNum").text.strip())

    link_list=[]    # 상품 링크를 추가할 리스트
    musinsa_data=[]
    link_num_list=[]
    # print(total_page_num)
    for page_num in range(1, total_page_num+1):
        print(page_num)
        # print(f"<<<<<<<<{page_num}페이지>>>>>>>>>")
        url=f"https://www.musinsa.com/app/styles/lists?&sort=NEWEST&page={page_num}"    # 쿠팡 제품 url 구조

        response=requests.get(url, headers=headers) 
        # 위의 url에 request를 보냄 -> 응답(상태코드, header, text, json 등의 웹페이지 정보)을 받음
        # 받은 응답에서 text를 지정(웹페이지 HTML 내용)
        html=response.text
        # 문자열로된 HTML을 HTML 요소를 검색, 조작할 수 있는 BeautifulSoup 객체로 변환 
        soup = BeautifulSoup(html, "html.parser")
        
        # 화면에 띄워진 제품들을 크롤링 (select함수 사용)
        items=soup.select("[class=style-list-item__thumbnail]")

        # 모든 제품 링크를 link_list 변수에 순서대로 저장
        for item in items:
            onclick_num=item.find("a")["onclick"]
            product_link_num = re.search(r"goView\('(\d+)'\)", onclick_num).group(1)
            link=f"https://www.musinsa.com/app/styles/views/{product_link_num}?"
            link_num_list.append(product_link_num)
            link_list.append(link)
        time.sleep(random.randint(1,2))
    print("제품 정보 크롤링 시작")
    for index, link in enumerate(link_list):
        response = requests.get(link, headers=headers)
        time.sleep(random.randint(1,2))
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        detail = soup.find_all('div', {'id': 'style_info'})
        for div in detail:
            item={}
            item['product_num']=link_num_list[index]
            # 코디 제목
            title = div.find('h2')
            if title:
                item['title']=title.text
            # 이미지 설명
            caption = div.find('p', {'class': 'styling_txt'})
            if caption:
                item['caption']=caption.text
            # 태그
            tags = div.find_all('a', {'class':'ui-tag-list__item'})
            tags_list=[]
            if(tags):
                for tag in tags:
                    tags_list.append(tag.text)
                item['tags']=tags_list
            # 코디 이미지
            images = div.find_all('img', {'class':'detail_img'})
            image_links = []
            for image in images:
                image_link = image.get('src')
                image_links.append(image_link)
            item['image_links'] = image_links
        musinsa_data.append(item)
        if(index%100==0):
            print(index, "번째 제품 크롤링 중")
        
    with open('musinsa_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(musinsa_data, json_file, ensure_ascii=False, indent=4)
    
    return 0


if __name__ == "__main__":
    musinsa_Crawling()