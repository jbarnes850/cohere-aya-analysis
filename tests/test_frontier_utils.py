from src.frontier_utils import coerce_lang, coerce_messages, detect_script_language, language_confused


def test_detect_script_language_ja():
    assert detect_script_language("これは日本語です") == "ja"


def test_detect_script_language_ko():
    assert detect_script_language("이 문장은 한국어입니다") == "ko"


def test_language_confused_for_ja_with_english():
    assert language_confused("ja", "This is English text")


def test_coerce_messages_from_instruction_output():
    row = {"instruction": "質問", "output": "回答"}
    messages = coerce_messages(row)
    assert messages is not None
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_coerce_messages_from_usr_bot_text():
    row = {"text": "<usr> 질문입니다\n<bot> 답변입니다"}
    messages = coerce_messages(row)
    assert messages is not None
    assert messages[0]["content"] == "질문입니다"
    assert messages[1]["content"] == "답변입니다"


def test_coerce_messages_from_inst_text():
    row = {"text": "<s>[INST] 翻訳してください [/INST] はい、翻訳します"}
    messages = coerce_messages(row)
    assert messages is not None
    assert messages[0]["content"] == "翻訳してください"
    assert messages[1]["content"] == "はい、翻訳します"


def test_coerce_lang_from_list_template_field():
    row = {"template_lang": ["jpn"]}
    assert coerce_lang(row) == "ja"
