import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """로거 설정을 위한 유틸리티 함수"""
    # 기존 로거가 있다면 반환
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 핸들러가 이미 있는지 확인
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 상위 로거로의 전파 방지
    logger.propagate = False
    
    return logger 