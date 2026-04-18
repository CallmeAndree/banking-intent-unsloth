import pandas as pd
import os
import re

id2label = {
    0: "activate_my_card", 1: "age_limit", 2: "apple_pay_or_google_pay", 3: "atm_support",
    4: "automatic_top_up", 5: "balance_not_updated_after_bank_transfer", 
    6: "balance_not_updated_after_cheque_or_cash_deposit", 7: "beneficiary_not_allowed",
    8: "cancel_transfer", 9: "card_about_to_expire", 10: "card_acceptance",
    11: "card_arrival", 12: "card_delivery_estimate", 13: "card_linking",
    14: "card_not_working", 15: "card_payment_fee_charged", 16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate", 18: "card_swallowed", 19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised", 21: "change_pin", 22: "compromised_card",
    23: "contactless_not_working", 24: "country_support", 25: "declined_card_payment",
    26: "declined_cash_withdrawal", 27: "declined_transfer", 28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits", 30: "edit_personal_details", 31: "exchange_charge",
    32: "exchange_rate", 33: "exchange_via_app", 34: "extra_charge_on_statement",
    35: "failed_transfer", 36: "fiat_currency_support", 37: "get_disposable_virtual_card",
    38: "get_physical_card", 39: "getting_spare_card", 40: "getting_virtual_card",
    41: "lost_or_stolen_card", 42: "lost_or_stolen_phone", 43: "order_physical_card",
    44: "passcode_forgotten", 45: "pending_card_payment", 46: "pending_cash_withdrawal",
    47: "pending_top_up", 48: "pending_transfer", 49: "pin_blocked", 50: "receiving_money",
    51: "Refund_not_showing_up", 52: "request_refund", 53: "reverted_card_payment?",
    54: "supported_cards_and_currencies", 55: "terminate_account", 
    56: "top_up_by_bank_transfer_charge", 57: "top_up_by_card_charge", 
    58: "top_up_by_cash_or_cheque", 59: "top_up_failed", 60: "top_up_limits",
    61: "top_up_reverted", 62: "topping_up_by_card", 63: "transaction_charged_twice",
    64: "transfer_fee_charged", 65: "transfer_into_account", 
    66: "transfer_not_received_by_recipient", 67: "transfer_timing", 
    68: "unable_to_verify_identity", 69: "verify_my_identity", 70: "verify_source_of_funds",
    71: "verify_top_up", 72: "virtual_card_not_working", 73: "visa_or_mastercard",
    74: "why_verify_identity", 75: "wrong_amount_of_cash_received", 
    76: "wrong_exchange_rate_for_cash_withdrawal"
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ các khoảng trắng ở đầu và cuối chuỗi
    text = text.strip()
    return text

def preprocess_data(input_path, output_path):
    print(f"Đang xử lý file: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_csv(input_path)
    
    # 1. Normalization text
    if 'text' in df.columns:
        df['text'] = df['text'].apply(normalize_text)
        print("- Đã chuẩn hóa cột 'text'.")
        
    # 2. Label mapping
    if 'label' in df.columns:
        # Tạo thêm cột 'label_name' bằng cách mapping từ id sang name
        df['label_name'] = df['label'].map(id2label)
        print("- Đã map label thành công.")
        
    # Lưu lại file đã qua xử lý
    df.to_csv(output_path, index=False)
    print(f"Đã lưu kết quả tại: {output_path}\n")

def main():
    data_dir = r"c:\NhutAnh\banking-intent-unsloth\sample_data"
    
    # File huấn luyện
    train_input = os.path.join(data_dir, "train.csv")
    train_output = os.path.join(data_dir, "train.csv")
    
    # File kiểm tra
    test_input = os.path.join(data_dir, "test.csv")
    test_output = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_input):
        preprocess_data(train_input, train_output)
    else:
        print(f"Không tìm thấy file: {train_input}")
        
    if os.path.exists(test_input):
        preprocess_data(test_input, test_output)
    else:
        print(f"Không tìm thấy file: {test_input}")

if __name__ == "__main__":
    main()
