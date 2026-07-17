class ToyTokenizer:
    TOKENS = ("中", "国", "首", "都", "是", "北", "京", "上", "海")
    LEGACY_IDS = (10, 20, 30, 40, 50, 60, 70, 80, 90)

    def __init__(self) -> None:
        self._token_to_id = {token: index for index, token in enumerate(self.TOKENS)}

    @property
    def vocab_size(self) -> int:
        return len(self.TOKENS)

    def encode(self, text: str) -> list[int]:
        result = []
        for position, token in enumerate(text):
            if token not in self._token_to_id:
                raise ValueError(f"未知字符 {token!r}，位置 {position}")
            result.append(self._token_to_id[token])
        return result

    def token(self, token_id: int) -> str:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"invalid token id {token_id}")
        return self.TOKENS[token_id]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token(token_id) for token_id in token_ids)

    def legacy_ids(self, text: str) -> list[int]:
        return [self.LEGACY_IDS[token_id] for token_id in self.encode(text)]
