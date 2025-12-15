import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class CBCTDataset(Dataset):
    def __init__(self,
                 patient_folders: list[str],        # список путей к папкам DICOM
                 mask_paths: list[str] = None,      # список путей к маскам (опционально)
                 target_spacing=(0.4, 0.4, 0.4),
                 clip_min=-1000, clip_max=3000,
                 transform=None):                   # опционально: аугментации
        """
        Конструктор: здесь инициализируем всё, что нужно для всех пациентов.
        Это вызывается один раз при создании датасета.
        """
        self.patient_folders = patient_folders
        self.mask_paths = mask_paths    # Разметка
        self.target_spacing = target_spacing
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.transform = transform  # например, MONAI transforms для аугментации

        # Можно здесь предвычислить что-то общее (например, статистики для нормализации)

    def __len__(self):
        """
        DataLoader использует это, чтобы знать, когда эпоха закончилась.
        """
        return len(self.patient_folders)

    def __getitem__(self, idx):
        """
        Обязательный метод: загрузка и предобработка ОДНОГО пациента по индексу.
        Вызывается каждый раз, когда DataLoader запрашивает следующий элемент.
        """
        # 1. Пути к данным текущего пациента
        dicom_folder = self.patient_folders[idx]
        patient_id = os.path.basename(dicom_folder.rstrip("/"))

        # 2. Загрузка изображения
        image_sitk = load_cbct_dicom_series(dicom_folder)
        image_tensor, metadata = preprocess_to_tensor(
            image_sitk,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            target_spacing=self.target_spacing
        )  # [1, D, H, W]

        # 3. Создаём словарь с данными
        sample = {
            "image": image_tensor,
            "patient_id": patient_id,
            "metadata": metadata
        }

        # 4. Если есть маски — загружаем и ресэмплим (с NearestNeighbor для меток!)
        # if self.mask_paths is not None:
        #     mask_sitk = sitk.ReadImage(self.mask_paths[idx])
        #     # Ресэмплим маску к тому же spacing и размеру, что и изображение
        #     # (отдельная функция resample_mask нужна)
        #     mask_resampled = resample_mask(mask_sitk, reference_image=image_sitk)
        #     mask_array = sitk.GetArrayFromImage(mask_resampled)
        #     mask_tensor = torch.from_numpy(mask_array).long().unsqueeze(0)  # [1, D, H, W]
        #     sample["mask"] = mask_tensor

        # 5. Опционально: аугментации (повороты, флипы и т.д.)
        if self.transform is not None:
            sample = self.transform(sample)  # MONAI или Albumentations

        return sample


def load_cbct_dicom_series(dicom_folder: str):
    """
    Читает всю серию DICOM-файлов из указанной папки и собирает их в один 3D-объём.
    Возвращает объект SimpleITK Image уже в единицах Хаунсфилда (HU).
    CBCT - Cone Beam Computed Tomography (конусно-лучевая компьютерная томография).
    """
    # Создаём объект для чтения серии DICOM
    reader = sitk.ImageSeriesReader()

    # Получаем список всех доступных серий в папке
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise ValueError("В папке не найдено DICOM-серий")

    # Выбираем самую большую серию по количеству файлов — это почти всегда нужная нам аксиальная серия КЛКТ
    largest_series = max(series_ids,
                         key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder, sid)))

    # Получаем список всех файлов, принадлежащих этой серии
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder, largest_series)

    # Указываем, какие файлы читать
    reader.SetFileNames(file_names)

    # Выполняем чтение — на выходе получаем 3D-изображение (sitk.Image)
    # Важно: SimpleITK автоматически применяет Rescale Slope и Rescale Intercept из DICOM-тегов
    # → значения уже в HU (Hounsfield Units) единицы Хаунсфилда.
    image = reader.Execute()

    return image


def preprocess_to_tensor(image: sitk.Image,
                         clip_min: float = -1000.0,                 # Нижняя граница клиппинга (воздух)
                         clip_max: float = 3000.0,                  # Верхняя граница (плотная кость/зубы)
                         target_spacing: tuple = (0.4, 0.4, 0.4),   # Желаемое изотропное разрешение в мм
):
    """
    Полная предобработка: клиппинг → ресэмплинг → нормализация → тензор PyTorch
    """
    # 1. Клиппинг значений HU — обрезаем экстремальные значения (особенно артефакты от металла)
    # Всё ниже -1000 становится -1000, всё выше 3000 становится 3000
    image = sitk.Clamp(image, lowerBound=clip_min, upperBound=clip_max)

    # 2. Ресэмплинг к единому (обычно изотропному) voxel spacing
    original_spacing = image.GetSpacing()   # Например: (0.3, 0.3, 0.5) мм
    original_size = image.GetSize()         # Например: (512, 512, 320)

    # Если текущее разрешение отличается от желаемого — делаем ресэмплинг
    if target_spacing != original_spacing:  # SimpleITK возвращает spacing в порядке (z,y,x)
        # Вычисляем новый размер объёма, чтобы сохранить физический размер объекта
        # Формула: new_size = old_size * old_spacing / new_spacing
        new_size = [
            int(round(orig_size * orig_sp / targ_sp))
            for orig_size, orig_sp, targ_sp in zip(original_size, original_spacing, target_spacing)
        ]
        new_size = [max(1, sz) for sz in new_size]

        # Настраиваем фильтр ресэмплинга
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)          # Берём ориентацию, origin и direction от исходного
        resampler.SetOutputSpacing(target_spacing)  # Новое разрешение (x, y, z)
        resampler.SetSize(new_size)                 # Новый размер в вокселях
        resampler.SetInterpolator(sitk.sitkLinear)  # Линейная интерполяция — оптимально для HU
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        # Выполняем ресэмплинг
        image = resampler.Execute(image)

    # 3. Преобразуем в numpy-массив
    # Важно: в SimpleITK массив имеет порядок (Z, Y, X), т.е. (глубина, высота, ширина)
    array = sitk.GetArrayFromImage(image)  # shape: (D, H, W)

    # 4. Нормализация в диапазон [0, 1]
    # Формула: (value - clip_min) / (clip_max - clip_min)
    # После клиппинга все значения гарантированно в [clip_min, clip_max]
    array = (array - clip_min) / (clip_max - clip_min)
    # Теперь воздух ≈ 0.0, мягкие ткани ≈ 0.5, кость ≈ 0.8–1.0

    # 5. Преобразуем в PyTorch-тензор и добавляем канал (для совместимости с 3D CNN)
    tensor = torch.from_numpy(array).float()  # shape: (D, H, W)
    tensor = tensor.unsqueeze(0)  # shape: (1, D, H, W) — канал первый (Channel-first)

    # 6. Сохраняем важные метаданные (понадобятся при постобработке и визуализации)
    metadata = {
        "spacing": target_spacing,
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "original_spacing": original_spacing
    }

    return tensor, metadata
