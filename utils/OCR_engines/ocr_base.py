# utils/ocr_base.py

# 파이썬의 추상 클래스(abstract base class)를 정의하기 위한 모듈가져오기
from abc import ABC, abstractmethod



# OCRBase라는 이름의 추상 클래스(ABC: Abstract Base Class)를 정의
# 이 클래스를 상속받는 하위 클래스들은 반드시 recognize_plate 메서드를 구현
class OCRBase(ABC):
     # 파이썬에서 메서드를 추상 메서드로 지정하기 위한 데코레이터
      # 하위 클래스는 이 메서드를 반드시 오버라이드
    @abstractmethod
    def recognize_plate(self, image_bytes: bytes) -> str:
        """
        이미지 바이트를 받아서 차량 번호판 문자열을 리턴합니다.
        (예: "12가3456"). 실패 시 빈 문자열("")을 리턴하거나 예외를 던집니다.
        """
        
         # 실제 구현은 하위 클래스에서 작성되어야 하므로, 여기서는 아무 동작도 하지않음
        pass
