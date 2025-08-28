def test_import():
    import genai_ingestion
    assert hasattr(genai_ingestion, "__all__")
