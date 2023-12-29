from unittest.mock import MagicMock, patch

import pytest

from whisperplus import LongTextSupportSummarizationPipeline


class TestLongTextSupportSummarizationPipeline:

    @pytest.fixture
    def summarizer(self):
        return LongTextSupportSummarizationPipeline()

    @pytest.fixture
    def summarizer_mocker(self):
        with patch(
                'whisperplus.pipelines.long_text_support_summarization.LongTextSupportSummarizationPipeline'
                '.load_model', new=lambda x: None):
            summarizer = LongTextSupportSummarizationPipeline()
            summarizer.load_model = MagicMock(side_effect=Exception("Model loading failed"))
            return summarizer

    def test_load_model(self, summarizer):
        # Test whether the model is loaded successfully
        assert summarizer.model is not None

    def test_split_text_into_chunks(self, summarizer):
        # Test chunking of splitting text
        text = "This is a test text. " * 50
        chunks = summarizer.split_text_into_chunks(text, 100)
        assert type(chunks) is list
        assert len(chunks) > 1  # 应该被分割成多个块

    def test_summarize_long_text_chunking(self, summarizer):
        # Test chunking of long text summaries
        long_text = "This is a long text. " * 1000
        summary = summarizer.summarize_long_text(long_text, 130, 30)
        assert type(summary) is str
        assert len(summary) > 0

    def test_summarize_short_text(self, summarizer):
        # Test summarization of short text
        short_text = "This is a short text."
        summary = summarizer.summarize(short_text)
        assert type(summary) is list
        assert len(summary) > 0

    def test_summarize_long_text(self, summarizer):
        # Test summarization of long text
        long_text = "This is a long text. " * 1000  # 创建一个足够长的文本
        summary = summarizer.summarize(long_text)
        assert type(summary) is str
        assert len(summary) > 0

    def test_model_loading_exception(self, summarizer_mocker):
        # Check if the model attribute is still None after an exception is raised
        assert summarizer_mocker.model is None

        with pytest.raises(Exception) as exc_info:
            summarizer_mocker.load_model()
            assert "Model loading failed" in str(exc_info.value)
