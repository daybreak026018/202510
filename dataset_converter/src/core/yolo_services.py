"""
Core services for the YOLO Studio workflow.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt", ".json", ".xml"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


@dataclass
class YOLOEnvironmentReport:
    python_executable: str
    environment_name: str
    python_version: str
    platform: str
    torch_installed: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str
    ultralytics_installed: bool
    ultralytics_version: str
    opencv_installed: bool
    opencv_version: str
    status: str
    cpu_install_command: str
    gpu_install_command: str


@dataclass
class YOLODatasetProfile:
    task: str
    dataset_path: str
    data_yaml: str
    classes: List[str]
    train_images: int
    val_images: int
    test_images: int
    label_files: int
    empty_labels: int
    orphan_images: int
    orphan_labels: int
    invalid_yolo_rows: int
    classification_classes: int
    warnings: List[str]


@dataclass
class YOLOTrainConfig:
    task: str
    model: str
    data_yaml: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    project: str
    name: str
    workers: int
    patience: int
    resume: bool
    python_executable: str = ""


@dataclass
class YOLOPredictConfig:
    task: str
    model_path: str
    source: str
    conf: float
    iou: float
    imgsz: int
    device: str
    save_txt: bool
    save_conf: bool
    output_dir: str
    python_executable: str = ""


class EnvironmentChecker:
    """Read-only dependency checker. It never installs packages."""

    @staticmethod
    def check(python_executable: str = None, environment_name: str = "当前环境") -> YOLOEnvironmentReport:
        python_executable = python_executable or sys.executable
        if Path(python_executable).resolve() != Path(sys.executable).resolve():
            return EnvironmentChecker._check_external_python(python_executable, environment_name)

        torch_installed = importlib.util.find_spec("torch") is not None
        torch_version = "未安装"
        cuda_available = False
        cuda_version = "不可用"
        gpu_name = "未检测到 GPU"

        if torch_installed:
            try:
                import torch

                torch_version = getattr(torch, "__version__", "未知")
                cuda_available = bool(torch.cuda.is_available())
                cuda_version = getattr(torch.version, "cuda", None) or "CPU"
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception as exc:
                torch_version = f"检测失败: {exc}"

        ultralytics_installed = importlib.util.find_spec("ultralytics") is not None
        ultralytics_version = "未安装"
        if ultralytics_installed:
            try:
                import ultralytics

                ultralytics_version = getattr(ultralytics, "__version__", "未知")
            except Exception as exc:
                ultralytics_version = f"检测失败: {exc}"

        opencv_installed = importlib.util.find_spec("cv2") is not None
        opencv_version = "未安装"
        if opencv_installed:
            try:
                import cv2

                opencv_version = getattr(cv2, "__version__", "未知")
            except Exception as exc:
                opencv_version = f"检测失败: {exc}"

        if not ultralytics_installed or not torch_installed:
            status = "缺依赖"
        elif cuda_available:
            status = "可训练-GPU"
        else:
            status = "可训练-CPU"

        return YOLOEnvironmentReport(
            python_executable=sys.executable,
            environment_name=environment_name,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
            torch_installed=torch_installed,
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            ultralytics_installed=ultralytics_installed,
            ultralytics_version=ultralytics_version,
            opencv_installed=opencv_installed,
            opencv_version=opencv_version,
            status=status,
            cpu_install_command="python -m pip install -U ultralytics torch torchvision torchaudio",
            gpu_install_command="请按 CUDA 版本到 PyTorch 官网选择 torch 安装命令，然后执行: python -m pip install -U ultralytics",
        )

    @staticmethod
    def _check_external_python(python_executable: str, environment_name: str) -> YOLOEnvironmentReport:
        script = (
            "import importlib.util,json,platform,sys;"
            "d={'python_version':sys.version.split()[0],'platform':platform.system()+' '+platform.release()};"
            "d['torch_installed']=importlib.util.find_spec('torch') is not None;"
            "d['ultralytics_installed']=importlib.util.find_spec('ultralytics') is not None;"
            "d['opencv_installed']=importlib.util.find_spec('cv2') is not None;"
            "d.update({'torch_version':'未安装','cuda_available':False,'cuda_version':'不可用','gpu_name':'未检测到 GPU','ultralytics_version':'未安装','opencv_version':'未安装'});"
            "\nif d['torch_installed']:\n"
            " import torch\n"
            " d['torch_version']=getattr(torch,'__version__','未知')\n"
            " d['cuda_available']=bool(torch.cuda.is_available())\n"
            " d['cuda_version']=getattr(torch.version,'cuda',None) or 'CPU'\n"
            " d['gpu_name']=torch.cuda.get_device_name(0) if d['cuda_available'] else d['gpu_name']\n"
            "if d['ultralytics_installed']:\n"
            " import ultralytics\n"
            " d['ultralytics_version']=getattr(ultralytics,'__version__','未知')\n"
            "if d['opencv_installed']:\n"
            " import cv2\n"
            " d['opencv_version']=getattr(cv2,'__version__','未知')\n"
            "print(json.dumps(d,ensure_ascii=False))"
        )
        try:
            proc = subprocess.run(
                [python_executable, "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=20,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "检测失败").strip())
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            if not data["ultralytics_installed"] or not data["torch_installed"]:
                status = "缺依赖"
            elif data["cuda_available"]:
                status = "可训练-GPU"
            else:
                status = "可训练-CPU"
            return YOLOEnvironmentReport(
                python_executable=python_executable,
                environment_name=environment_name,
                python_version=data["python_version"],
                platform=data["platform"],
                torch_installed=data["torch_installed"],
                torch_version=data["torch_version"],
                cuda_available=data["cuda_available"],
                cuda_version=data["cuda_version"],
                gpu_name=data["gpu_name"],
                ultralytics_installed=data["ultralytics_installed"],
                ultralytics_version=data["ultralytics_version"],
                opencv_installed=data["opencv_installed"],
                opencv_version=data["opencv_version"],
                status=status,
                cpu_install_command=f"\"{python_executable}\" -m pip install -U ultralytics torch torchvision torchaudio",
                gpu_install_command=f"请按 CUDA 版本安装匹配 torch，然后执行: \"{python_executable}\" -m pip install -U ultralytics",
            )
        except Exception as exc:
            return YOLOEnvironmentReport(
                python_executable=python_executable,
                environment_name=environment_name,
                python_version="检测失败",
                platform=platform.system(),
                torch_installed=False,
                torch_version=f"检测失败: {exc}",
                cuda_available=False,
                cuda_version="不可用",
                gpu_name="未检测到 GPU",
                ultralytics_installed=False,
                ultralytics_version="检测失败",
                opencv_installed=False,
                opencv_version="检测失败",
                status="环境异常",
                cpu_install_command=f"\"{python_executable}\" -m pip install -U ultralytics torch torchvision torchaudio",
                gpu_install_command=f"请先确认该 Python 可执行文件有效: {python_executable}",
            )


class CondaEnvironmentManager:
    """Discover local Conda environments without activating shells."""

    @staticmethod
    def list_environments() -> List[Dict[str, str]]:
        envs = [{"name": "当前Python", "path": str(Path(sys.executable).parent), "python": sys.executable}]
        conda_exe = CondaEnvironmentManager._find_conda_exe()
        if conda_exe:
            envs.extend(CondaEnvironmentManager._list_from_conda(conda_exe))
        envs.extend(CondaEnvironmentManager._list_from_common_dirs())
        return CondaEnvironmentManager._dedupe_envs(envs)

    @staticmethod
    def _find_conda_exe() -> Optional[str]:
        candidates = ["conda"]
        import os

        if os.environ.get("CONDA_EXE"):
            candidates.insert(0, os.environ["CONDA_EXE"])
        python_dir = Path(sys.executable).parent
        for conda_exe in (
            python_dir / "Scripts" / "conda.exe",
            python_dir.parent / "Scripts" / "conda.exe",
            python_dir.parent.parent / "Scripts" / "conda.exe" if len(python_dir.parents) > 1 else None,
        ):
            if conda_exe and conda_exe.exists():
                candidates.insert(0, str(conda_exe))
        for candidate in candidates:
            try:
                proc = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return candidate
            except Exception:
                continue
        return None

    @staticmethod
    def _list_from_conda(conda_exe: str) -> List[Dict[str, str]]:
        try:
            proc = subprocess.run(
                [conda_exe, "env", "list", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=10,
            )
            if proc.returncode != 0:
                return []
            data = json.loads(proc.stdout)
            return [CondaEnvironmentManager._env_from_path(Path(path)) for path in data.get("envs", [])]
        except Exception:
            return []

    @staticmethod
    def _list_from_common_dirs() -> List[Dict[str, str]]:
        roots = [
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path("D:/Users/lst/miniconda3"),
            Path("D:/Users/lst/anaconda3"),
            Path("C:/Users/lst/miniconda3"),
            Path("C:/Users/lst/anaconda3"),
        ]
        envs = []
        for root in roots:
            if root.exists():
                envs.append(CondaEnvironmentManager._env_from_path(root))
                env_dir = root / "envs"
                if env_dir.exists():
                    envs.extend(CondaEnvironmentManager._env_from_path(path) for path in env_dir.iterdir() if path.is_dir())
        return envs

    @staticmethod
    def _env_from_path(path: Path) -> Dict[str, str]:
        python = path / "python.exe" if platform.system() == "Windows" else path / "bin" / "python"
        name = "base" if path.name.lower() in {"miniconda3", "anaconda3"} else path.name
        return {"name": name, "path": str(path), "python": str(python)}

    @staticmethod
    def _dedupe_envs(envs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        deduped = []
        seen = set()
        for env in envs:
            python = env.get("python", "")
            if not python or python in seen or not Path(python).exists():
                continue
            seen.add(python)
            deduped.append(env)
        return deduped


class YOLODatasetInspector:
    """Profile YOLO detection, segmentation, and classification datasets."""

    def inspect(self, dataset_path: Path, task: str = "detect") -> YOLODatasetProfile:
        dataset_path = Path(dataset_path)
        if task == "classify":
            return self._inspect_classification(dataset_path)
        return self._inspect_detection_like(dataset_path, task)

    def generate_data_yaml(self, dataset_path: Path, task: str, class_names: Optional[List[str]] = None) -> Path:
        dataset_path = Path(dataset_path)
        names = class_names or self._read_class_names(dataset_path) or ["class0"]
        data = {
            "path": str(dataset_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names,
        }
        target = dataset_path / "data.yaml"
        target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return target

    def _inspect_detection_like(self, dataset_path: Path, task: str) -> YOLODatasetProfile:
        data_yaml = dataset_path / "data.yaml"
        classes = self._read_class_names(dataset_path)
        train_images = self._count_images(dataset_path / "images" / "train")
        val_images = self._count_images(dataset_path / "images" / "val")
        test_images = self._count_images(dataset_path / "images" / "test")

        image_files = self._collect_images(dataset_path / "images")
        label_files = [p for p in (dataset_path / "labels").rglob("*.txt")] if (dataset_path / "labels").exists() else []
        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}
        empty_labels = sum(1 for p in label_files if p.stat().st_size == 0)
        invalid_rows = sum(self._count_invalid_yolo_rows(p, task) for p in label_files if p.stat().st_size > 0)

        warnings = []
        if not data_yaml.exists():
            warnings.append("缺少 data.yaml，可在数据准备页生成。")
        if not classes:
            warnings.append("未读取到类别名称，建议准备 classes.txt 或 data.yaml。")
        if not image_files:
            warnings.append("未发现 images 目录下的图像文件。")
        if not label_files:
            warnings.append("未发现 labels 目录下的 YOLO 标注。")
        if empty_labels:
            warnings.append(f"发现 {empty_labels} 个空标注文件，请确认是否为负样本。")
        if invalid_rows:
            warnings.append(f"发现 {invalid_rows} 行疑似异常 YOLO 标注。")

        return YOLODatasetProfile(
            task=task,
            dataset_path=str(dataset_path),
            data_yaml=str(data_yaml) if data_yaml.exists() else "",
            classes=classes,
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            label_files=len(label_files),
            empty_labels=empty_labels,
            orphan_images=len(image_stems - label_stems),
            orphan_labels=len(label_stems - image_stems),
            invalid_yolo_rows=invalid_rows,
            classification_classes=0,
            warnings=warnings,
        )

    def _inspect_classification(self, dataset_path: Path) -> YOLODatasetProfile:
        split_counts = {}
        class_names = set()
        warnings = []
        for split in ("train", "val", "test"):
            split_dir = dataset_path / split
            count = self._count_images(split_dir)
            split_counts[split] = count
            if split_dir.exists():
                class_names.update(p.name for p in split_dir.iterdir() if p.is_dir())
        if not split_counts["train"]:
            warnings.append("分类数据集缺少 train/类别名/图片 结构。")
        if not class_names:
            warnings.append("未发现分类类别目录。")

        return YOLODatasetProfile(
            task="classify",
            dataset_path=str(dataset_path),
            data_yaml="分类任务通常直接使用数据集根目录",
            classes=sorted(class_names),
            train_images=split_counts["train"],
            val_images=split_counts["val"],
            test_images=split_counts["test"],
            label_files=0,
            empty_labels=0,
            orphan_images=0,
            orphan_labels=0,
            invalid_yolo_rows=0,
            classification_classes=len(class_names),
            warnings=warnings,
        )

    def _read_class_names(self, dataset_path: Path) -> List[str]:
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
                names = data.get("names", [])
                if isinstance(names, dict):
                    return [names[key] for key in sorted(names)]
                if isinstance(names, list):
                    return [str(item) for item in names]
            except Exception:
                pass
        for name in ("classes.txt", "labels.txt"):
            path = dataset_path / name
            if path.exists():
                return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return []

    def _collect_images(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    def _count_images(self, root: Path) -> int:
        return len(self._collect_images(root))

    def _count_invalid_yolo_rows(self, path: Path, task: str) -> int:
        invalid = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            min_len = 5 if task == "detect" else 7
            if len(parts) < min_len:
                invalid += 1
                continue
            try:
                values = [float(item) for item in parts[1:]]
            except ValueError:
                invalid += 1
                continue
            if any(value < 0 or value > 1 for value in values):
                invalid += 1
        return invalid


class YOLOCommandBuilder:
    @staticmethod
    def build_train(config: YOLOTrainConfig) -> List[str]:
        data_arg = config.data_yaml if config.task == "classify" else config.data_yaml
        runner_path = Path(__file__).resolve().parent / "yolo_cli_runner.py"
        args = [
            config.python_executable or sys.executable,
            str(runner_path),
            "train",
            "--task",
            config.task,
            "--model",
            config.model,
            "--data",
            data_arg,
            "--epochs",
            str(config.epochs),
            "--imgsz",
            str(config.imgsz),
            "--batch",
            str(config.batch),
            "--device",
            config.device,
            "--project",
            config.project,
            "--name",
            config.name,
            "--workers",
            str(config.workers),
            "--patience",
            str(config.patience),
        ]
        if config.resume:
            args.append("--resume")
        return args

    @staticmethod
    def build_predict(config: YOLOPredictConfig) -> List[str]:
        runner_path = Path(__file__).resolve().parent / "yolo_cli_runner.py"
        args = [
            config.python_executable or sys.executable,
            str(runner_path),
            "predict",
            "--task",
            config.task,
            "--model",
            config.model_path,
            "--source",
            config.source,
            "--conf",
            str(config.conf),
            "--iou",
            str(config.iou),
            "--imgsz",
            str(config.imgsz),
            "--device",
            config.device,
            "--project",
            config.output_dir,
            "--name",
            "predict",
        ]
        if config.save_txt:
            args.append("--save-txt")
        if config.save_conf:
            args.append("--save-conf")
        return args


class RunsManager:
    def __init__(self, root: Path):
        self.root = Path(root)

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if not self.root.exists():
            return runs
        for path in sorted([p for p in self.root.rglob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            best = path / "weights" / "best.pt"
            last = path / "weights" / "last.pt"
            curve = self._find_curve_image(path)
            preview_images = self._find_preview_images(path)
            if best.exists() or last.exists() or curve or preview_images:
                runs.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "type": self._guess_type(path),
                        "best": str(best) if best.exists() else "",
                        "last": str(last) if last.exists() else "",
                        "curve": str(curve) if curve else "",
                        "preview_images": [str(item) for item in preview_images],
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        return runs

    def save_run_record(self, record: Dict[str, str]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / "yolo_studio_runs.json"
        records = []
        if target.exists():
            try:
                records = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                records = []
        records.insert(0, record)
        target.write_text(json.dumps(records[:50], ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _find_curve_image(self, path: Path) -> Optional[Path]:
        for name in ("results.png", "results.jpg", "results.jpeg"):
            candidate = path / name
            if candidate.exists():
                return candidate
        return None

    def _find_preview_images(self, path: Path) -> List[Path]:
        image_files = []
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "weights" in candidate.parts:
                continue
            image_files.append(candidate)
        image_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return image_files[:24]

    def _guess_type(self, path: Path) -> str:
        lowered = str(path).lower()
        if "predict" in lowered:
            return "检测"
        if "train" in lowered:
            return "训练"
        return "结果"
