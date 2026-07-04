from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):

    name: str = 'Ayush'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt= 0, lt=10, description="CGPA must be between 0 and 10")

new_student = {'age': '24', 'email': 'ayush@example.com', 'cgpa': '8.5'}

st = Student(**new_student)

print(st)