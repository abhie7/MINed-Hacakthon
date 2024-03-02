from parsers import PdfParser, PptxParser, DocxParser

class MainParser:
    '''Class for extracting text from different file formats.'''
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.extension = filepath.split('.')[-1] # extract the file extension

    def extract_text(self) -> str:
        # create the appropriate parser based on the file extension and call its method to get the text
        if self.extension == 'pdf':
            extractor = PdfParser(self.filepath)
        elif self.extension == 'pptx':
            extractor = PptxParser(self.filepath)
        elif self.extension == 'docx':
            extractor = DocxParser(self.filepath)
        else:
            # raise an error if the file extension is not supported
            raise ValueError(f'Invalid file extension: {self.extension}')
        return extractor.extract_text() # return the extracted text

# test
if __name__ == "__main__":
    # filepath = "./data/files/Desires.pdf"
    # filepath = "./data/files/Presentation.pptx"
    filepath = "./data/files/Lab4.docx"
    parser = MainParser(filepath)
    extracted_text = parser.extract_text()
    print(extracted_text)