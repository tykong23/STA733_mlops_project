import streamlit as st
import pandas as pd
from utils import *

st.sidebar.title("Dashboard")
option = st.sidebar.selectbox("섹션 선택", ["전체 상품 목록","재고 추가", "재고 현황 및 주요 지표", "안전 재고 및 재주문점 계산", "수요 예측 결과", "재고 시뮬레이션"])

if option == "전체 상품 목록":
    categories = pd.read_csv('data/categories.csv')
    st.title('전체 상품 코드와 카테고리')
    st.write('총 312개 상품')
    st.write('카테고리 : 식품, 화장품/미용, 패션잡화, 출산/육아, 생활건강')
    st.dataframe(categories.iloc[:,1:])
 
if option == "재고 추가":
    st.title('입/출고 입력')
    st.header('입/출고 데이터 추가')
    csv_choice = st.selectbox('입고/출고 선택', ['출고', '입고'])

    if csv_choice == '출고':
        st.subheader("출고 데이터 입력")
        i = st.text_input("상품 코드 (i)")
        t = st.text_input("날짜 (t, 예: 2024-06-21)")
        y = st.number_input("수량 (y)", step=1.0)
        if st.button('출고에 데이터 추가'):
            new_data = {'i': [i], 't': [t], 'y': [y]}
            updated_demand_df = add_data_to_csv(demand_file_path, new_data)
            st.write("출고에 데이터가 성공적으로 추가되었습니다.")
            st.write(updated_demand_df.tail())
            stock_df = update_stock_csv()
            st.write("재고가 성공적으로 업데이트되었습니다.")
    else:
        st.subheader("입고 데이터 입력")
        i = st.text_input("상품 코드 (i)")
        t = st.text_input("날짜 (t, 예: 2024-06-21)")
        y = st.number_input("수량 (y)", step=1.0)
        if st.button('입고에 데이터 추가'):
            new_data = {'i': [i], 't': [t], 'y': [y]}
            updated_inventory_df = add_data_to_csv(inventory_file_path, new_data)
            st.write("입고에 데이터가 성공적으로 추가되었습니다.")
            st.write(updated_inventory_df.tail())
            stock_df = update_stock_csv()
            st.write("재고가 성공적으로 업데이트되었습니다.")

elif option == "재고 현황 및 주요 지표":
    st.title('재고 현황 및 주요 지표')

    # 전체 현재 재고량
    st.header('전체 현재 재고량')
    stock_df = pd.read_csv(stock_file_path)
    stock_df['t'] = pd.to_datetime(stock_df['t'])
    stock_df['i'] = stock_df['i'].astype(str)

    # 각 상품별로 가장 최신 날짜의 재고 데이터만 선택
    latest_stock = stock_df.sort_values('t').groupby('i').tail()

    fig = px.bar(latest_stock, x='y', y='i', title='상품별 현재 재고량', orientation='h',
                 labels={'i': '상품 ID', 'y': '현재 재고량'})
    st.plotly_chart(fig)

    # 지정 상품의 현재 재고량
    st.header('지정 상품의 현재 재고량')
    items = st.text_input('상품 ID 입력 (예: 10011,730211)')
    if st.button('재고량 확인'):
        items_list = [item.strip() for item in items.split(',')]
        filtered_stock = latest_stock[latest_stock['i'].isin(items_list)]
        filtered_stock = filtered_stock[['i', 'y']]
        filtered_stock.columns = ['상품', '현재 재고량']

        fig = px.bar(filtered_stock, x='현재 재고량', y='상품', title=f'상품 {items}의 현재 재고량', orientation='h')
        st.plotly_chart(fig)

    # 상품의 실시간 재고 수준 추이
    st.header('상품의 실시간 재고 수준 추이')
    product_id = st.text_input('상품 ID 입력 (예: 10011)')
    if st.button('실시간 재고 추이 확인'):
        product_df = stock_df[stock_df['i'] == product_id]
        fig = px.line(product_df, x='t', y='y', title=f'상품 : {product_id} 재고 수준 추이',
                      labels={'t': '날짜', 'y': '재고 수준'})
        st.plotly_chart(fig)

elif option == "안전 재고 및 재주문점 계산":
    st.title('안전 재고 및 재주문점 계산')

    st.header('안전 재고 및 재주문점 계산')

    col1, col2 = st.columns(2)

    with col1:
        product_id = st.text_input('상품 ID 입력 (예: 10011)')
    with col2:
        service_level = st.number_input('서비스 수준 입력 (예: 0.9 = 90%, 0.95 = 95%)', min_value=0.01, max_value=0.99, value=0.95, step=0.01)

    if st.button('계산'):
        if product_id and service_level:
            demand_df = pd.read_csv(demand_file_path)
            inventory_df = pd.read_csv(inventory_file_path)
            lead_time_df = pd.read_csv(lead_time_file_path)

            unique_lead_time_df = lead_time_df[['상품코드_WMS_', 'average_lead_time']].drop_duplicates()
            unique_lead_time_df['average_lead_time'] = pd.to_timedelta(unique_lead_time_df['average_lead_time'])
            inventory_df['입고량_pcs_'] = inventory_df['y']

            if 'y' in demand_df.columns and 'y' in inventory_df.columns:
                try:
                    avg_lead_time = unique_lead_time_df.set_index('상품코드_WMS_').loc[product_id, 'average_lead_time']
                    order_quantity_mean = np.round(inventory_df[inventory_df['i'] == product_id]['y'].mean())
                    
                    safety_stock = calc_safety_stock(demand_df[demand_df['i'] == product_id], avg_lead_time, service_level)
                    reorder_point = calc_reorder_point(demand_df[demand_df['i'] == product_id], avg_lead_time, safety_stock)

                    st.header(f'상품 ID {product_id}의 안전 재고')
                    st.write(safety_stock)

                    st.header(f'상품 ID {product_id}의 재주문점')
                    st.write(reorder_point)
                except KeyError:
                    st.write(f"상품 ID {product_id}에 대한 리드 타임 데이터를 찾을 수 없습니다.")
            else:
                st.write("리드 타임 및 입고량에 대한 필요한 컬럼이 CSV 파일에 없습니다.")
        else:
            st.write("상품 ID와 서비스 수준을 입력해주세요.")

elif option == "수요 예측 결과":
    st.title('수요 예측 결과')

    product_id_input = st.text_input('상품 ID 입력 (예: 10011)')

    if st.button('예측 결과 확인'):
        product_id = str(product_id_input).strip()

        if product_id:
            st.header(f'Prophet 예측 결과 - 상품 ID: {product_id}')
            predict_and_plot_prophet(product_id)
            
            st.header(f'LightGBM 예측 결과 - 상품 ID: {product_id}')
            demand_df = pd.read_csv(demand_file_path)
            demand_df['t'] = pd.to_datetime(demand_df['t'])
            predict_and_plot_lightgbm(product_id, demand_df)
        else:
            st.write("상품 ID를 입력해주세요.")

elif option == "재고 시뮬레이션":
    st.title('재고 시뮬레이션')

    product_id = st.text_input('상품 ID 입력 (예: 10011)')
    
    if st.button('시뮬레이션 시작'):
        if product_id:
            simulate_stock(product_id)
        else:
            st.write("상품 ID를 입력해주세요.")