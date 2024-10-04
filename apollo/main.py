import numpy as np
from apollo.student_grade import *


def main():
	prediction_service = PredictionService()
	prediction_service.train_model()

	new_student = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3])

	predicted_grade = prediction_service.predict_grade(new_student)
	print(f'Grade: {predicted_grade}')


if __name__ == '__main__':
	main()
