from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionTemplate:
    instruction_prefix: str = "Instruction:\n"
    response_prefix: str = "\n\nResponse:\n"

    def prompt_text(self, instruction: str) -> str:
        instruction = " ".join(instruction.strip().split())
        if not instruction:
            raise ValueError("instruction must not be empty")
        return f"{self.instruction_prefix}{instruction}{self.response_prefix}"


DEFAULT_TEMPLATE = DiffusionTemplate()
