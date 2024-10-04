import numpy as np
from sklearn.neural_network import MLPRegressor
from abc import ABC, abstractmethod

class GradePredictor(ABC):
	"""
	Абстрактный базовый класс для предсказания оценок студентов.
	"""
	@abstractmethod
	def train(self, x_train, y_train):
		"""
		Обучает модель на предоставленных данных.

		Args:
			x_train (numpy.ndarray): Входные данные для обучения.
			y_train (numpy.ndarray): Ожидаемые выходы (оценки) для обучения.
		"""
		pass

	@abstractmethod
	def predict(self, x):
		"""
		Предсказывает оценки для входных данных.

		Args:
			x (numpy.ndarray): Входные данные для предсказания.

		Returns:
			numpy.ndarray: Предсказанные оценки.
		"""
		pass


class NeuralNetworkGradePredictor(GradePredictor):
	"""
	Реализация предсказания оценок с использованием нейронной сети.
	"""
	def __init__(self, hidden_layer_sizes=(1000, 990, 900), max_iter=1000000, random_state=42):
		"""
		Инициализирует модель нейронной сети.

		Args:
			hidden_layer_sizes (tuple): Количество нейронов в скрытых слоях.
			max_iter (int): Максимальное количество итераций обучения.
			random_state (int): Состояние генератора случайных чисел для воспроизводимости.
		"""
		self.model = self._create_model(hidden_layer_sizes, max_iter, random_state)

	def _create_model(self, hidden_layer_sizes, max_iter, random_state):
		"""
		Создает модель нейронной сети.

		Args:
			hidden_layer_sizes (tuple): Количество нейронов в скрытых слоях.
			max_iter (int): Максимальное количество итераций обучения.
			random_state (int): Состояние генератора случайных чисел для воспроизводимости.

		Returns:
			sklearn.neural_network.MLPRegressor: Модель нейронной сети.
		"""
		return MLPRegressor(
			hidden_layer_sizes=hidden_layer_sizes,
			activation='relu',
			solver='adam',
			max_iter=max_iter,
			random_state=random_state
		)

	def train(self, x_train, y_train):
		"""
		Обучает модель на предоставленных данных.

		Args:
			x_train (numpy.ndarray): Входные данные для обучения.
			y_train (numpy.ndarray): Ожидаемые выходы (оценки) для обучения.
		"""
		self.model.fit(x_train, y_train)

	def predict(self, x):
		"""
		Предсказывает оценки для входных данных.

		Args:
			x (numpy.ndarray): Входные данные для предсказания.

		Returns:
			numpy.ndarray: Предсказанные оценки.
		"""
		return self.model.predict(x)

class DataLoader:
	"""
	Класс для загрузки и подготовки тренировочных данных.
	"""
	@staticmethod
	def load_training_data():
		"""
		Загружает и возвращает тренировочные данные.

		Returns:
			tuple: Кортеж, содержащий входные данные (X) и ожидаемые выходы (y).
		"""
		x_train = np.array([
			[0.8, 0.7, 0.9, 0.6, 0.3, 4.5],
			[0.6, 0.8, 0.7, 0.8, 0.4, 4.0],
			[0.9, 0.6, 0.8, 0.7, 0.2, 4.7],
			[0.7, 0.7, 0.6, 0.9, 0.5, 4.2],
			[0.75, 0.75, 0.75, 0.75, 0.25, 4.3],
			[0.65, 0.85, 0.65, 0.85, 0.35, 4.6],
			[0.85, 0.55, 0.85, 0.55, 0.15, 4.8],
			[0.6, 0.8, 0.6, 0.9, 0.4, 4.1],

			[0.7, 0.6, 0.8, 0.8, 0.3, 4.4],
			[0.8, 0.7, 0.7, 0.7, 0.2, 4.6],
			[0.9, 0.8, 0.6, 0.6, 0.4, 4.2],
			[0.6, 0.7, 0.8, 0.8, 0.3, 4.5],
			[0.7, 0.8, 0.7, 0.7, 0.2, 4.3],
			[0.8, 0.6, 0.8, 0.7, 0.3, 4.7],
			[0.7, 0.7, 0.7, 0.8, 0.4, 4.1],
			[0.75, 0.65, 0.85, 0.75, 0.15, 4.9],
			[0.65, 0.75, 0.75, 0.85, 0.25, 4.2],
			[0.8, 0.8, 0.6, 0.7, 0.3, 4.4],
			[0.7, 0.7, 0.8, 0.6, 0.4, 4.6],
			[0.85, 0.65, 0.75, 0.75, 0.2, 4.8],
			[0.7, 0.8, 0.75, 0.75, 0.25, 4.3],
			[0.75, 0.75, 0.8, 0.7, 0.2, 4.5],
			[0.8, 0.7, 0.75, 0.75, 0.3, 4.7],
			[0.65, 0.8, 0.7, 0.8, 0.35, 4.1],
			[0.75, 0.75, 0.75, 0.75, 0.25, 4.4],
			[0.55, 0.6, 0.7, 0.6, 0.7, 2],
			[0.45, 0.5, 0.6, 0.5, 0.8, 2],
			[0.65, 0.6, 0.65, 0.65, 0.5, 3],
			[0.6, 0.55, 0.7, 0.6, 0.6, 3],
			[0.5, 0.5, 0.6, 0.6, 0.7, 1]
		])

		y_train = np.array([4, 4, 5, 4, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 4, 5, 4, 4, 2, 2, 3, 3, 1])

		return x_train, y_train

class PredictionService:
	"""
	Сервис для предсказания оценок студентов.
	"""
	def __init__(self):
		self.grade_predictor = NeuralNetworkGradePredictor()

	def train_model(self):
		"""
		Тренирует модель на данных.
		"""
		x_train, y_train = DataLoader.load_training_data()
		self.grade_predictor.train(x_train, y_train)

	def predict_grade(self, student_data):
		"""
		Предсказывает оценку для данных студента.

		Args:
			student_data (numpy.ndarray): Характеристики студента.

		Returns:
			float: Предсказанная оценка.
		"""
		return self.grade_predictor.predict([student_data])[0]

	def get_feature_importances(self):
		"""
		Возвращает важность каждого признака для предсказания оценки.

		Returns:
			numpy.ndarray: Массив важности признаков.
		"""
		return self.grade_predictor.model.feature_importances_
