import pandas as pd
import glob
import os

# 엑셀 파일 디렉토리 경로
input_dir = 'data/custom/data/'
output_excel_path = 'data/custom/result/output_file.csv'

# 디렉토리 내의 모든 .xlsx 파일 찾기
xlsx_files = glob.glob(os.path.join(input_dir, '*.xlsx'))

if not xlsx_files:
    print("No Excel files found in the directory.")
else:
    # 첫 번째 .xlsx 파일 선택
    input_excel_path = xlsx_files[0]

    # 엑셀 파일 읽기
    df = pd.read_excel(input_excel_path)

    # 엑셀 파일의 열 이름 출력
    print("Columns in the Excel file:", df.columns.tolist())

    # 'prediction' 열을 'ko' 열로 대체하고 나머지 열 제거
    if 'prediction' in df.columns:
        df = df.rename(columns={'prediction': 'ko'})[['ko']]

        # 결과를 엑셀 파일로 저장
        df.to_csv(output_excel_path, index=False)


        print(f"Processed data saved to {output_excel_path}")
    else:
        print("'prediction' column not found in the Excel file.")
