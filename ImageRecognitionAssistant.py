import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image, ImageDraw, ImageFont
import io
import json
import openai
import requests
import base64
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'subscription_status' not in st.session_state:
    st.session_state.subscription_status = None

if 'users' not in st.session_state:
    st.session_state.users = {}

if 'show_payment_page' not in st.session_state:
    st.session_state.show_payment_page = False

API_KEY = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
USER_DATA_FILE = 'users.json'
SERVICE_ACCOUNT_FILE = 'service_account.json'

USAGE_LIMIT = 10

def main():
    load_users()
    if st.session_state.logged_in:
        if st.session_state.show_payment_page:
            show_payment_page()
        else:
            show_main_page()
    else:
        login_or_register()

def show_main_page():
    st.title("圖片辨識助手")

    st.sidebar.title("使用次數")
    if st.session_state.subscription_status:
        st.sidebar.write(f"已使用 {st.session_state.usage_count} 次，訂閱用戶無限制")
    else:
        st.sidebar.write(f"已使用 {st.session_state.usage_count} 次, 免費使用上限 {USAGE_LIMIT} 次")

    if st.sidebar.button("返回主界面", key="back_to_main"):
        st.session_state.logged_in = False
        st.experimental_rerun()
        return

    st.sidebar.title("當前使用模型")
    current_model = "GPT-4o" if st.session_state.subscription_status else "GPT-3.5"
    st.sidebar.write(f"目前使用的模型是：{current_model}")

    st.sidebar.title("辨識歷史記錄")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history):
            if st.sidebar.button(f"查看記錄 {idx + 1}", key=f"view_record_{idx}"):
                st.sidebar.image(entry['image'], caption=f"記錄 {idx + 1}: {entry['result']}", use_column_width=True)
                st.sidebar.write(entry['description'])
    else:
        st.sidebar.write("尚未有辨識記錄")

    if st.session_state.subscription_status:
        if st.sidebar.button("取消訂閱", key="cancel_subscription"):
            st.session_state.subscription_status = False
            st.session_state.users[st.session_state.current_user]['subscription_status'] = False
            save_users()
            st.success("您已取消訂閱，現為免費用戶")
            st.experimental_rerun()
            return
    else:
        if st.sidebar.button("訂閱", key="subscribe"):
            st.session_state.show_payment_page = True
            st.experimental_rerun()
            return

    st.write("請上傳一張圖片，我會嘗試辨識它並提供相關信息。")

    # Check usage limit
    if st.session_state.usage_count >= USAGE_LIMIT and not st.session_state.subscription_status:
        st.error("您已達到免費使用次數上限，請訂閱來繼續使用。")
        if st.button("訂閱", key="subscribe_limited"):
            st.session_state.show_payment_page = True
            st.experimental_rerun()
        return

    uploaded_file = st.file_uploader("選擇圖片...", type=["jpg", "jpeg", "png", "webp", "bmp"], key="file_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='上傳的圖片', use_column_width=True)
        st.write("")

        # Select detection type
        detection_type = st.selectbox("選擇辨識類型", ["標籤辨識", "網頁辨識", "物體辨識", "OCR文字辨識", "Logo辨識", "不當內容辨識"], key="detection_type")

        # Generate response
        if st.button("辨識圖片", key="detect_image"):
            with st.spinner("正在辨識..."):
                if detection_type == "標籤辨識":
                    result, image_with_result = label_detection(image)
                elif detection_type == "網頁辨識":
                    result, image_with_result = web_detection(image)
                elif detection_type == "物體辨識":
                    result, image_with_result = object_detection(image)
                elif detection_type == "OCR文字辨識":
                    result, image_with_result = ocr_detection(image)
                elif detection_type == "Logo辨識":
                    result, image_with_result = logo_detection(image)
                elif detection_type == "不當內容辨識":
                    result, image_with_result = explicit_content_detection(image)

                st.session_state.usage_count += 1 
                st.write(result, unsafe_allow_html=True)
                st.image(image_with_result, caption='辨識結果', use_column_width=True)
            try:
                description = generate_gpt_description(result, st.session_state.subscription_status)
                st.write("生成的描述：")
                st.write(description)
            except Exception as e:
                st.error(f"生成描述失敗：{str(e)}")
                description = "生成描述失敗"

            buffered = io.BytesIO()

            if image_with_result.mode == 'RGBA':
                image_with_result = image_with_result.convert('RGB')

            image_with_result.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()
            st.session_state.history.append({"image": image_bytes, "result": result, "description": description})
            save_users() 
            st.experimental_rerun()

def login_or_register():
    st.title("歡迎使用圖片辨識助手")
    choice = st.sidebar.selectbox("選擇操作", ["登錄", "註冊"], key="login_or_register")

    if choice == "登錄":
        st.subheader("登錄")
        username = st.text_input("用戶名", key="login_username")
        password = st.text_input("密碼", type="password", key="login_password")
        if st.button("登錄", key="login_button"):
            if username in st.session_state.users and st.session_state.users[username]['password'] == password:
                st.success("登錄成功")
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.usage_count = st.session_state.users[username]['usage_count']
                st.session_state.subscription_status = st.session_state.users[username].get('subscription_status', False)
                st.experimental_rerun()
            else:
                st.error("用戶名或密碼錯誤")
    elif choice == "註冊":
        st.subheader("註冊")
        new_username = st.text_input("用戶名", key="new_username")
        new_password = st.text_input("密碼", type="password", key="new_password")
        if st.button("註冊", key="register_button"):
            if new_username in st.session_state.users:
                st.error("用戶名已存在")
            else:
                st.session_state.users[new_username] = {'password': new_password, 'usage_count': 0, 'subscription_status': False}
                save_users()
                st.success("註冊成功，請登錄")
                st.session_state.logged_in = False
                st.experimental_rerun()

def label_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始標籤辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.label_detection(image=image_vision)
        labels = response.label_annotations

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "標籤辨識結果：<br>"
        for label in labels:
            result += f"{label.description}: {label.score*100:.2f}%<br>"
            logging.info(f"標籤：{label.description}，信心度：{label.score*100:.2f}%")

        return result, image

    except Exception as e:
        logging.error(f"標籤辨識失敗：{str(e)}")
        return f"標籤辨識失敗：{str(e)}", image

def web_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始網頁辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.web_detection(image=image_vision)
        annotations = response.web_detection

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "網頁辨識結果：<br>"
        if annotations.pages_with_matching_images:
            result += "匹配的網頁：<br>"
            for page in annotations.pages_with_matching_images:
                result += f"{page.url}<br>"
                logging.info(f"匹配的網頁：{page.url}")

        return result, image

    except Exception as e:
        logging.error(f"網頁辨識失敗：{str(e)}")
        return f"網頁辨識失敗：{str(e)}", image

def object_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始物體辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.object_localization(image=image_vision)
        objects = response.localized_object_annotations

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "物體辨識結果：<br>"
        for object_ in objects:
            result += f"{object_.name}: {object_.score*100:.2f}%<br>"
            logging.info(f"物體：{object_.name}，信心度：{object_.score*100:.2f}%")

            box = [(vertex.x * image.width, vertex.y * image.height) for vertex in object_.bounding_poly.normalized_vertices]
            draw.polygon(box, outline='red')
            draw.text(box[0], object_.name, fill='red', font=font)

        return result, image

    except Exception as e:
        logging.error(f"物體辨識失敗：{str(e)}")
        return f"物體辨識失敗：{str(e)}", image

def ocr_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始OCR文字辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.text_detection(image=image_vision)
        texts = response.text_annotations

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "OCR文字辨識結果：<br>"
        for text in texts:
            result += f"{text.description}<br>"
            logging.info(f"OCR文字：{text.description}")

        return result, image

    except Exception as e:
        logging.error(f"OCR文字辨識失敗：{str(e)}")
        return f"OCR文字辨識失敗：{str(e)}", image

def logo_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始Logo辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.logo_detection(image=image_vision)
        logos = response.logo_annotations

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "Logo辨識結果：<br>"
        for logo in logos:
            result += f"{logo.description}: {logo.score*100:.2f}%<br>"
            logging.info(f"Logo：{logo.description}，信心度：{logo.score*100:.2f}%")

        return result, image

    except Exception as e:
        logging.error(f"Logo辨識失敗：{str(e)}")
        return f"Logo辨識失敗：{str(e)}", image

def explicit_content_detection(image):
    try:
        def get_image_content(image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            content = buffered.getvalue()
            return content

        logging.info("開始不當內容辨識")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = get_image_content(image)
        image_vision = vision.Image(content=content)
        response = client.safe_search_detection(image=image_vision)
        safe = response.safe_search_annotation

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        result = "不當內容辨識結果：<br>"
        result += f"成人內容：{safe.adult}<br>"
        result += f"暴力內容：{safe.violence}<br>"
        result += f"醫療內容：{safe.medical}<br>"
        result += f"情色內容：{safe.racy}<br>"
        logging.info(f"不當內容：成人-{safe.adult}, 暴力-{safe.violence}, 醫療-{safe.medical}, 情色-{safe.racy}")

        return result, image

    except Exception as e:
        logging.error(f"不當內容辨識失敗：{str(e)}")
        return f"不當內容辨識失敗：{str(e)}", image

def generate_gpt_description(result, use_gpt4=False):
    try:
        if use_gpt4:
            model = "gpt-4o"
            messages = [
                {"role": "system", "content": "你是專業圖片描述生成助手,以繁體中文回答,請確實地描述圖片狀況,不要用記錄呈現的文字回答"},
                {"role": "user", "content": f"根據以下辨識結果生成一段描述：{result}"}
            ]
        else:
            model = "gpt-3.5-turbo"
            messages = [
                {"role": "system", "content": "你是專業圖片描述生成助手，以繁體中文回答，並且作簡短回覆就好，不要用記錄呈現的文字回答"},
                {"role": "user", "content": f"根據以下辨識結果生成一段描述：{result}"}
            ]
        
        logging.info(f"使用模型 {model} 生成描述")
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=200
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"生成描述錯誤: {str(e)}")
        return f"生成描述失敗: {str(e)}"
    
def load_users():
    try:
        with open(USER_DATA_FILE, 'r') as file:
            st.session_state.users = json.load(file)
    except FileNotFoundError:
        st.session_state.users = {}

def save_users():
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(st.session_state.users, file)

def get_image_content(image):
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        content = output.getvalue()
    return content

def show_payment_page():
    if st.session_state.get('subscription_status'):
        show_success_page()
        return
    
    st.title("訂閱付款")
    st.write("請選擇訂閱計劃並完成付款以獲得無限次使用次數。")
    
    with st.form("payment_form"):
        st.write("請輸入信用卡信息以完成付款，買斷制價格500台幣：")
        card_number = st.text_input("信用卡號碼")
        card_expiry = st.text_input("到期日 (MM/YY)")
        card_cvc = st.text_input("CVC")
        
        submit_payment = st.form_submit_button("付款")
        cancel_payment = st.form_submit_button("取消付款")

        if submit_payment:
            if card_number and card_expiry and card_cvc:
                st.session_state.subscription_status = True
                st.session_state.users[st.session_state.current_user]['subscription_status'] = True
                save_users()
                st.experimental_rerun()  # 重新加載頁面以顯示成功頁面
            else:
                st.error("請填寫所有信用卡信息")
        
        if cancel_payment:
            st.session_state.show_payment_page = False
            st.experimental_rerun()

def show_success_page():
    st.title("訂閱成功")
    st.write("訂閱成功！請繼續體驗無限制的辨識功能與更強大的模型功能!")
if __name__ == "__main__":
    main()
