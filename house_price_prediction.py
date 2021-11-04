import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
#from lightgbm import LGBMRegressor
from PIL import Image

# ---Part 1: Load data and model-------------------------------------------------------
data = pd.read_excel('products_EDA.xlsx')
addresses = pd.read_excel('addresses.xlsx')

# Load model
with open('HousePriceScaler.pkl', 'rb') as file:
  hpscaler = pickle.load(file)
  
with open('HousePriceModel.pkl', 'rb') as file:
  hpmodel = pickle.load(file)

# ---Part 2: Show project result with Streamlit------------------------------------------
st.title('Đồ Án Tốt Nghiệp Data Science')
st.header('Dự đoán giá nhà riêng tại TP HCM')

# Tạo menu
menu = ['Giới thiệu', 'Xây dựng mô hình', 'Dự đoán giá trị mới']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Giới thiệu':
# Hiển thị giới thiệu chung
    st.subheader("Giới thiệu")
    st.write("""
    #### 
    - Mục tiêu của đồ án là xây dựng một ứng dụng dự đoán giá nhà riêng tại TP Hồ Chí Minh, hỗ trợ cho người dùng đang có nhu cầu mua nhà, bán nhà hoặc cán bộ Tổ chức tín dụng cần thẩm định giá trị tài sản bảo đảm là bất động sản
    - Mô hình dự đoán sử dụng dữ liệu huấn luyện được lấy từ từ trang web https://propzy.vn/mua/nha/hcm - là trang môi giới mua bán bất động sản với phân khúc nhà riêng đang rao bán tại TP Hồ Chí Minh.
    - Để sử dụng ứng dụng, người dùng nhập các thông tin mô tả căn nhà đang có nhu cầu mua / bán / thẩm định giá, ứng dụng sẽ trả về mức giá dự đoán tương ứng.
    - Đồ án được thực hiện bởi học viên Nguyễn Thùy Linh với sự hướng dẫn, hỗ trợ của giảng viên, thầy giáo Nguyễn Quan Liêm - TTTH - ĐHKHTN - Trường DDHQG TP HCM  
    """)

elif choice == 'Xây dựng mô hình':
# Xây dựng mô hình
    st.subheader('Xây dựng mô hình')
    # Giới thiệu output

    # Hiển thị dữ liệu
    st.write('#### Dữ liệu & biến được chọn')
    st.table(data.head(5))
    st.markdown('''
    - **Output** 
      - HV lựa chọn mô hình dự đoán giá nhà / m2 (price_avg_pre) thay vì mô hình dự đoán giá nhà
      - Giá nhà được tính = Giá nhà / m2 * diện tích đất
    - **Input** - Các biến được lựa chọn đưa vào mô hình bao gồm:
      - **price_unit_pre**: Giá đất  / m2 trung bình theo khu vực - là biến được tổng hợp và tính toán lại từ các biến **district** (Quận), **ward** (Phường), **street** (Đường)
      - **floor**: Số tầng của căn nhà
      - **s_ground_pre**: diện tích đất
      - **alley_road_width**: độ rộng đường trước nhà
      - **road_type**: đường trước nhà là đường hay hẻm (1-Đường, 0-Hẻm)
      - **rooftop**: Có / không có tầng thường (1-Có, 0-Không)
      - **near_market**: Có / không gần chợ (1-Có, 0-Không)
    ''')

    # Tham số đánh giá
    st.write('#### Tham số đánh giá')
    metrics = [81.55, 80.59, 550.04, 23.45]
    metrics = pd.DataFrame(metrics, index=['Score Train', 'Score Test', 'MSE', 'RMSE'], columns=['LGBMRegressor'])
    st.table(metrics.head(5))
    st.write('**Ghi chú**: RMSE 23,45 triệu tương đương khoảng 20% giá nhà / m2 trung bình của toàn bộ tập dữ liệu')

    # Trực quan hóa kết quả
    st.write('#### Trực quan hóa kết quả')
    image = Image.open('scatterplot.png')
    st.image(image)

elif choice == 'Dự đoán giá trị mới':
    # Người dùng nhập thông tin nhà
    st.subheader("Dự đoán giá trị mới")
    st.write("#### Nhập vào dữ liệu")
    districts = np.sort(addresses['district'].unique())
    district = st.selectbox('Quận',  options=districts)
    wards = np.sort(addresses[addresses['district']==district]['ward'].unique())
    ward = st.selectbox('Phường',  options=wards)
    streets = np.sort(addresses[(addresses['district']==district) & (addresses['ward']==ward)]['street'].unique())
    street = st.selectbox('Đường',  options=streets)
    s_ground_pre = st.number_input('Diện tích')
    floor = st.slider('Số tầng', 1.0, 7.0, 2.0, 0.5)
    rooftop_str = st.radio('Nhà có sân thượng không?', options=['Có', 'Không'])
    road_type_str = st.radio('Trước nhà là đường hay hẻm?', options=['Đường', 'Hẻm'])
    alley_road_width = st.slider('Độ rộng đường trước nhà', 0.94, 10.5, 3.0)
    near_market_str = st.radio('Nhà có gần chợ trong bán kính 500m?', options=['Có', 'không'])
    #st.write(district, ward, street)

    # Make new prediction
    if st.button('Tính giá nhà'):
      price_unit_pre = addresses[(addresses['district']==district) &\
         (addresses['ward']==ward) & (addresses['street']==street)]['price_unit_pre'].mean()
      if rooftop_str == 'Có':
        rooftop = 1
      else:
        rooftop = 0
      if road_type_str == 'Đường':
        road_type = 1
      else:
        road_type = 0
      if near_market_str == 'Có':
        near_market = 1
      else:
        near_market = 0
      X_new = [[rooftop,near_market,road_type,price_unit_pre,floor,s_ground_pre,alley_road_width]]
      X_new = hpscaler.transform(X_new)
      y_new = hpmodel.predict(X_new)[0]
      price = round(y_new * s_ground_pre / 1000,2)
      st.write('#### Giá trị căn nhà: {} tỷ VND'.format(price))
