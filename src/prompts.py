"""Prompt suite for multilingual analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional


LANGUAGES = ["en", "ja", "ko", "zh", "vi", "id", "th"]

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
}


@dataclass
class Prompt:
    text: str
    language: str
    category: str
    expected_next: Optional[str] = None
    expected_concept: Optional[str] = None


TECHNICAL = {
    "en": Prompt(
        text="The server configuration requires port 443 for HTTPS. The database connection string is",
        language="en", category="technical",
        expected_concept="connection_string",
    ),
    "ja": Prompt(
        text="サーバー設定にはHTTPS用のポート443が必要です。データベース接続文字列は",
        language="ja", category="technical",
        expected_concept="connection_string",
    ),
    "ko": Prompt(
        text="서버 구성에는 HTTPS용 포트 443이 필요합니다. 데이터베이스 연결 문자열은",
        language="ko", category="technical",
        expected_concept="connection_string",
    ),
    "zh": Prompt(
        text="服务器配置需要端口443用于HTTPS。数据库连接字符串是",
        language="zh", category="technical",
        expected_concept="connection_string",
    ),
    "vi": Prompt(
        text="Cấu hình máy chủ yêu cầu cổng 443 cho HTTPS. Chuỗi kết nối cơ sở dữ liệu là",
        language="vi", category="technical",
        expected_concept="connection_string",
    ),
    "id": Prompt(
        text="Konfigurasi server memerlukan port 443 untuk HTTPS. String koneksi database adalah",
        language="id", category="technical",
        expected_concept="connection_string",
    ),
    "th": Prompt(
        text="การกำหนดค่าเซิร์ฟเวอร์ต้องการพอร์ต 443 สำหรับ HTTPS สตริงการเชื่อมต่อฐานข้อมูลคือ",
        language="th", category="technical",
        expected_concept="connection_string",
    ),
}

NAMED_ENTITY = {
    "en": Prompt(
        text="Fujitsu Limited announced a partnership with",
        language="en", category="named_entity",
        expected_concept="company_name",
    ),
    "ja": Prompt(
        text="富士通株式会社は提携を発表しました。相手は",
        language="ja", category="named_entity",
        expected_concept="company_name",
    ),
    "ko": Prompt(
        text="삼성전자는 파트너십을 발표했습니다. 상대는",
        language="ko", category="named_entity",
        expected_concept="company_name",
    ),
    "zh": Prompt(
        text="华为技术有限公司宣布与以下公司建立合作伙伴关系：",
        language="zh", category="named_entity",
        expected_concept="company_name",
    ),
    "vi": Prompt(
        text="Công ty Fujitsu đã công bố hợp tác với",
        language="vi", category="named_entity",
        expected_concept="company_name",
    ),
    "id": Prompt(
        text="Fujitsu Limited mengumumkan kemitraan dengan",
        language="id", category="named_entity",
        expected_concept="company_name",
    ),
    "th": Prompt(
        text="บริษัท ฟูจิตสึ จำกัด ประกาศความร่วมมือกับ",
        language="th", category="named_entity",
        expected_concept="company_name",
    ),
}

ANTONYM = {
    "en": Prompt(text="The opposite of 'large' is '", language="en", category="antonym",
                 expected_next="small", expected_concept="antonym_large"),
    "ja": Prompt(text="「大きい」の反対は「", language="ja", category="antonym",
                 expected_next="小さい", expected_concept="antonym_large"),
    "ko": Prompt(text="'크다'의 반대말은 '", language="ko", category="antonym",
                 expected_next="작다", expected_concept="antonym_large"),
    "zh": Prompt(text="'大'的反义词是'", language="zh", category="antonym",
                 expected_next="小", expected_concept="antonym_large"),
    "vi": Prompt(text="Từ trái nghĩa của 'lớn' là '", language="vi", category="antonym",
                 expected_next="nhỏ", expected_concept="antonym_large"),
    "id": Prompt(text="Lawan kata dari 'besar' adalah '", language="id", category="antonym",
                 expected_next="kecil", expected_concept="antonym_large"),
    "th": Prompt(text="คำตรงข้ามของ 'ใหญ่' คือ '", language="th", category="antonym",
                 expected_next="เล็ก", expected_concept="antonym_large"),
}

ANTONYM_HOT = {
    "en": Prompt(text="The opposite of 'hot' is '", language="en", category="antonym_hot",
                 expected_next="cold", expected_concept="antonym_hot"),
    "ja": Prompt(text="「暑い」の反対は「", language="ja", category="antonym_hot",
                 expected_next="寒い", expected_concept="antonym_hot"),
    "ko": Prompt(text="'덥다'의 반대말은 '", language="ko", category="antonym_hot",
                 expected_next="춥다", expected_concept="antonym_hot"),
    "zh": Prompt(text="'热'的反义词是'", language="zh", category="antonym_hot",
                 expected_next="冷", expected_concept="antonym_hot"),
    "vi": Prompt(text="Từ trái nghĩa của 'nóng' là '", language="vi", category="antonym_hot",
                 expected_next="lạnh", expected_concept="antonym_hot"),
    "id": Prompt(text="Lawan kata dari 'panas' adalah '", language="id", category="antonym_hot",
                 expected_next="dingin", expected_concept="antonym_hot"),
    "th": Prompt(text="คำตรงข้ามของ 'ร้อน' คือ '", language="th", category="antonym_hot",
                 expected_next="เย็น", expected_concept="antonym_hot"),
}

ANTONYM_FAST = {
    "en": Prompt(text="The opposite of 'fast' is '", language="en", category="antonym_fast",
                 expected_next="slow", expected_concept="antonym_fast"),
    "ja": Prompt(text="「速い」の反対は「", language="ja", category="antonym_fast",
                 expected_next="遅い", expected_concept="antonym_fast"),
    "ko": Prompt(text="'빠르다'의 반대말은 '", language="ko", category="antonym_fast",
                 expected_next="느리다", expected_concept="antonym_fast"),
    "zh": Prompt(text="'快'的反义词是'", language="zh", category="antonym_fast",
                 expected_next="慢", expected_concept="antonym_fast"),
    "vi": Prompt(text="Từ trái nghĩa của 'nhanh' là '", language="vi", category="antonym_fast",
                 expected_next="chậm", expected_concept="antonym_fast"),
    "id": Prompt(text="Lawan kata dari 'cepat' adalah '", language="id", category="antonym_fast",
                 expected_next="lambat", expected_concept="antonym_fast"),
    "th": Prompt(text="คำตรงข้ามของ 'เร็ว' คือ '", language="th", category="antonym_fast",
                 expected_next="ช้า", expected_concept="antonym_fast"),
}

ANTONYM_GOOD = {
    "en": Prompt(text="The opposite of 'good' is '", language="en", category="antonym_good",
                 expected_next="bad", expected_concept="antonym_good"),
    "ja": Prompt(text="「良い」の反対は「", language="ja", category="antonym_good",
                 expected_next="悪い", expected_concept="antonym_good"),
    "ko": Prompt(text="'좋다'의 반대말은 '", language="ko", category="antonym_good",
                 expected_next="나쁘다", expected_concept="antonym_good"),
    "zh": Prompt(text="'好'的反义词是'", language="zh", category="antonym_good",
                 expected_next="坏", expected_concept="antonym_good"),
    "vi": Prompt(text="Từ trái nghĩa của 'tốt' là '", language="vi", category="antonym_good",
                 expected_next="xấu", expected_concept="antonym_good"),
    "id": Prompt(text="Lawan kata dari 'baik' adalah '", language="id", category="antonym_good",
                 expected_next="buruk", expected_concept="antonym_good"),
    "th": Prompt(text="คำตรงข้ามของ 'ดี' คือ '", language="th", category="antonym_good",
                 expected_next="ไม่ดี", expected_concept="antonym_good"),
}

ANTONYM_LONG = {
    "en": Prompt(text="The opposite of 'long' is '", language="en", category="antonym_long",
                 expected_next="short", expected_concept="antonym_long"),
    "ja": Prompt(text="「長い」の反対は「", language="ja", category="antonym_long",
                 expected_next="短い", expected_concept="antonym_long"),
    "ko": Prompt(text="'길다'의 반대말은 '", language="ko", category="antonym_long",
                 expected_next="짧다", expected_concept="antonym_long"),
    "zh": Prompt(text="'长'的反义词是'", language="zh", category="antonym_long",
                 expected_next="短", expected_concept="antonym_long"),
    "vi": Prompt(text="Từ trái nghĩa của 'dài' là '", language="vi", category="antonym_long",
                 expected_next="ngắn", expected_concept="antonym_long"),
    "id": Prompt(text="Lawan kata dari 'panjang' adalah '", language="id", category="antonym_long",
                 expected_next="pendek", expected_concept="antonym_long"),
    "th": Prompt(text="คำตรงข้ามของ 'ยาว' คือ '", language="th", category="antonym_long",
                 expected_next="สั้น", expected_concept="antonym_long"),
}

ANTONYM_HIGH = {
    "en": Prompt(text="The opposite of 'high' is '", language="en", category="antonym_high",
                 expected_next="low", expected_concept="antonym_high"),
    "ja": Prompt(text="「高い」の反対は「", language="ja", category="antonym_high",
                 expected_next="低い", expected_concept="antonym_high"),
    "ko": Prompt(text="'높다'의 반대말은 '", language="ko", category="antonym_high",
                 expected_next="낮다", expected_concept="antonym_high"),
    "zh": Prompt(text="'高'的反义词是'", language="zh", category="antonym_high",
                 expected_next="低", expected_concept="antonym_high"),
    "vi": Prompt(text="Từ trái nghĩa của 'cao' là '", language="vi", category="antonym_high",
                 expected_next="thấp", expected_concept="antonym_high"),
    "id": Prompt(text="Lawan kata dari 'tinggi' adalah '", language="id", category="antonym_high",
                 expected_next="rendah", expected_concept="antonym_high"),
    "th": Prompt(text="คำตรงข้ามของ 'สูง' คือ '", language="th", category="antonym_high",
                 expected_next="ต่ำ", expected_concept="antonym_high"),
}

STRUCTURED_DATA = {
    "en": Prompt(
        text="In 2024, revenue was 500 million yen. In 2025 it was",
        language="en", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "ja": Prompt(
        text="2024年の売上は5億円でした。2025年の売上は",
        language="ja", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "ko": Prompt(
        text="2024년 매출은 5억 엔이었습니다. 2025년 매출은",
        language="ko", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "zh": Prompt(
        text="2024年营收为5亿日元。2025年营收为",
        language="zh", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "vi": Prompt(
        text="Năm 2024, doanh thu là 500 triệu yên. Năm 2025 là",
        language="vi", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "id": Prompt(
        text="Pada tahun 2024, pendapatan adalah 500 juta yen. Pada tahun 2025 adalah",
        language="id", category="structured_data",
        expected_concept="numeric_continuation",
    ),
    "th": Prompt(
        text="ในปี 2024 รายได้คือ 500 ล้านเยน ในปี 2025 คือ",
        language="th", category="structured_data",
        expected_concept="numeric_continuation",
    ),
}

DAY_SEQUENCE = {
    "en": Prompt(
        text="Monday, Tuesday, Wednesday, Thursday,",
        language="en", category="day_sequence",
        expected_next="Friday",
        expected_concept="day_sequence",
    ),
    "ja": Prompt(
        text="月曜日、火曜日、水曜日、木曜日、",
        language="ja", category="day_sequence",
        expected_next="金曜日",
        expected_concept="day_sequence",
    ),
    "ko": Prompt(
        text="월요일, 화요일, 수요일, 목요일,",
        language="ko", category="day_sequence",
        expected_next="금요일",
        expected_concept="day_sequence",
    ),
    "zh": Prompt(
        text="星期一、星期二、星期三、星期四、",
        language="zh", category="day_sequence",
        expected_next="星期五",
        expected_concept="day_sequence",
    ),
    "vi": Prompt(
        text="Thứ hai, thứ ba, thứ tư, thứ năm,",
        language="vi", category="day_sequence",
        expected_next="thứ sáu",
        expected_concept="day_sequence",
    ),
    "id": Prompt(
        text="Senin, Selasa, Rabu, Kamis,",
        language="id", category="day_sequence",
        expected_next="Jumat",
        expected_concept="day_sequence",
    ),
    "th": Prompt(
        text="วันจันทร์ วันอังคาร วันพุธ วันพฤหัสบดี",
        language="th", category="day_sequence",
        expected_next="วันศุกร์",
        expected_concept="day_sequence",
    ),
}

ALL_CATEGORIES = {
    "technical": TECHNICAL,
    "named_entity": NAMED_ENTITY,
    "antonym": ANTONYM,
    "antonym_hot": ANTONYM_HOT,
    "antonym_fast": ANTONYM_FAST,
    "antonym_good": ANTONYM_GOOD,
    "antonym_long": ANTONYM_LONG,
    "antonym_high": ANTONYM_HIGH,
    "structured_data": STRUCTURED_DATA,
    "day_sequence": DAY_SEQUENCE,
}


def get_all_prompts(languages: Optional[List[str]] = None) -> List[Prompt]:
    """Return flat list of prompts, optionally filtered by language."""
    if languages is None:
        languages = LANGUAGES
    prompts = []
    for category_prompts in ALL_CATEGORIES.values():
        for lang, prompt in category_prompts.items():
            if lang in languages:
                prompts.append(prompt)
    return prompts


def get_prompts_by_category(category: str, languages: Optional[List[str]] = None) -> Dict[str, Prompt]:
    """Return prompts for a single category, keyed by language."""
    if languages is None:
        languages = LANGUAGES
    return {
        lang: prompt
        for lang, prompt in ALL_CATEGORIES[category].items()
        if lang in languages
    }


ENTERPRISE_ENTITIES = {
    "en": ["Fujitsu", "Toyota", "Samsung", "LG Electronics", "Hyundai", "Huawei", "Tencent"],
    "ja": ["富士通", "トヨタ", "サムスン", "LGエレクトロニクス", "ヒュンダイ", "ファーウェイ", "テンセント"],
    "ko": ["후지쯔", "도요타", "삼성", "LG전자", "현대", "화웨이", "텐센트"],
    "zh": ["富士通", "丰田", "三星", "LG电子", "现代", "华为", "腾讯"],
}

MIXED_SCRIPT_STRINGS = [
    "HTTPSにはポート443が必要です",
    "API키를 설정하세요",
    "使用SSL证书配置HTTPS",
    "Cấu hình HTTPS với port 443",
    "Konfigurasi HTTPS pada port 443",
    "ตั้งค่า HTTPS บนพอร์ต 443",
]
