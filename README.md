
# RetargetVid & SmartVidCrop

**Languages / 语言：** [English](#introduction) | [简体中文](#简体中文)

## Introduction
This repository contains **RetargetVid** - a video retargeting dataset, and **SmartVidCrop** - a fast cropping method for video retargeting, i.e. for transforming a video's aspect ratio.

We observed that each literature work about video retargeting uses an arbitrary selection of videos to test its results, and most of the time these videos are not provided, while the evaluation procedure relies on visual inspection of selected frames. Motivated by this we construct and release RetargetVid, a publicly available benchmark dataset for video cropping, annotated by 6 human subjects.

We also release a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step.

## RetargetVid Dataset
We selected a subset of 200 videos from the publicly available videos of the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset, specifically, the first 100 videos of the training set (videos with indices 001 to 100) and the 100 videos of the validation set (videos with indices 601 to 700). All videos are in 16:9 aspect ratio and most of them consist of a single shot.

We invited 6 human subjects and asked them to select the region of each frame that would be ideal to be included in a cropped version of the video. Specifically, we assigned them the task of generating two cropped versions for each video, one with target aspect ratio of 1:3 and another one with target aspect ratio of 3:1. We selected these extreme target aspect ratios (despite not being used in real-life applications) in order to identify human preferences under very demanding circumstances. Moreover, less extreme target aspect ratios can still be evaluated by assessing to what extent an e.g. 9:16 crop window includes the 1:3 manually specified window.

Our crop window annotations for each video are in the form of text files, where the *i*-th line contains the top-left and bottom-right coordinates of the crop window for the *i*-th frame. The coordinates are zero-based, so the top-left pixel of a frame is the (0,0). There are 2400 annotation text files in total (200 videos * 2 target aspect ratios * 6 annotator subjects). The annotation text files are named *$video_id$-$target_aspect ratio$.txt*, where *$video_id$* is the original video filename and $target_aspect ratio$ is the target aspect ratio (i.e. "1-3" or "3-1"). The annotation text files can be found in the *annotations* directory of this repository, where a separate zip file is provided for each annotator. To download the actual DHF1k videos, follow the download links in the original author's GitHub repository [here](https://github.com/wenguanwang/DHF1K).


## SmartVidCrop Method
We argue that cropping methods are more suitable for video aspect ratio transformation when the minimization of semantic distortions is a prerequisite. Therefore, we present a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step. For our method, we utilize visual saliency to find the image regions of attention, and we employ a filtering-through-clustering technique to select the main region of focus. For more details, see the first citation in the Citations section.

Our method is implemented in Python 3 and the main script is the *smartVidCrop.py* of this repository. You can import this to your own python scripts or run it. Running it will invoke the main method and re-produce the results of our paper. 

We have also implemented a technique to assess the quality of the cropped version and opt to resort to padding in the case where the results of cropping are unsatisfactory due to the nature of the video’s content, i.e. only a small percentage of the salient visual information can be covered by any crop window. This is due to acknowledging that a cropped version of a video cannot always retain all regions of interest of the original video. This feature is disabled by default. If you want to enable this feature as well as adjust other parameters of our method, see "sc_init_crop_params" function in the *smartVidCrop.py*.


## Evaluation & Updated SmartVidCrop Method
To evaluate the results of your method with respect to the ground truth annotations of the **RetargetVid** dataset, first download this repository. Then, create a new sub-directory in the *results* directory, in which a text file for each of the 200 videos of the dataset must be created. The files must follow the naming convention *$video_id$-$target_aspect_ratio$.txt*, where *$video_id$* is the original video filename and $target_aspect_ratio$ is the target aspect ratio (i.e. "1-3" or "3-1"). Note that our smartVidCrop method can transform a video to any target aspect ratio, however our evaluation script is a bit more restrictive and can only evaluate results in "1:3" and "3:1" target aspect ratios. Each line of this text file must have the crop window (top, left, bottom, right) coordinates. Finally, run the *retargetvid_eval.py* python script. The evaluation results for every sub-directory in the *results* directory will be displayed, warning you if there were any errors in the process or any incomplete annotations were found.

The evaluation results are calculated as the mean similarity of all crop windows between the results contained in a sub-directory of the *results* directory and the RetargetVid dataset's ground-truth annotations. The similarity is calculated in terms of the Intersection over Union (IoU) scores.

In the *results* directory we include two sub-directories:
* *smartvidcrop* with the results of our method, and
* *autoflip* with the results of Google's [AutoFlip](https://google.github.io/mediapipe/solutions/autoflip) method,
for you to quickly replicate the results of our paper (see the first citation in the Citations section).
 
The software was implemented in Python 3 and the source code is included in the *retargetvid_eval.py* file of this repository.

After publishing our IEEE ICIP 2021 paper we introduced four additional changes in our algorithm, to further improve the results reported in this paper. The improved algorithm is presented in our IEEE ISM 2021 paper. Specifically, the updates reported in our ISM 2021 paper are as follows: First, to speed-up the retargeting process we spatially sub-sample the saliency maps that are fed into the filtering-through-clustering procedure. The second update is the introduction of a "focus stability" mechanism in order to reject sudden changes in focus, while the third update is the replacement of the LOESS method employed in the ICIP 2021 paper for time-series smoothing with a Savitzky-Golay filter, since visual inspection of the results for both approaches revealed that the latter yields a smoother crop window movement. Finally, we employed an optimized set of parameters. To deploy all of the aforementioned updates, call the *sc_init_crop_params* method changing the optional argument *use_best_settings* to True. The default settings of the *smartVidCrop.py* script will reproduce the results of our ICIP paper. In the following table we report the resulting gain in terms of average IoU across all 6 annotators of the RetargetVid dataset ("mean IoU"; higher values are better) and the ratio of execution time in our test PC to video duration ("t"; lower values are better, and a value lower than 100% indicates faster-than-realtime processing).

<table>
<thead>
  <tr>
    <th rowspan="2"><b>parameters set</b></th>
    <th colspan="2" style="text-align:center;"><b>1:3 aspect ratio</b></th>
    <th colspan="2" style="text-align:center;"><b>3:1 aspect ratio</b></th>
  </tr>
  <tr>
    <td style="text-align:center;"><b>mean<br>IoU</b></td>
    <td style="text-align:center;"><b>t</b></td>
    <td style="text-align:center;"><b>mean<br>IoU</b></td>
    <td style="text-align:center;"><b>t</b></td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ICIP 2021 paper settings (default)</td>
    <td style="text-align:center;">49.9%</td>
    <td style="text-align:center;">19%</td>
    <td style="text-align:center;">71.4%</td>
    <td style="text-align:center;">20%</td>
  </tr>
  <tr>
    <td>ISM 2021 paper settings (by changing the optional argument *use_best_settings* to True)</td>
   <td style="text-align:center;">52.9%</td>
    <td style="text-align:center;">13%</td>
    <td style="text-align:center;">75.3%</td>
    <td style="text-align:center;">14%</td>
  </tr>
</tbody>
</table>

## Basketball Vertical Video Studio

This repository also includes a local browser MVP for converting basketball footage
from landscape to portrait while keeping a selected player in the crop:

1. Install the product dependencies into the repository's Python environment:

   ```powershell
   .\venv37\Scripts\python.exe -m pip install -r requirements-product.txt
   ```

2. Start the local application and check the optional YOLO/CUDA environment:

   ```powershell
   .\venv37\Scripts\python.exe .\start_basketball.py
   ```

3. Open <http://127.0.0.1:8765> and upload a landscape video. The app estimates
   hard cuts from color-histogram changes and divides the footage into shot
   segments. Select one segment, choose a clear frame within it, and click
   **Analyze current frame** to view ranked player candidates. Select a candidate
   to confirm it, or drag a box to override the suggestion. Then click
   **Lock target and generate tracking preview** and scrub the timeline to inspect
   the tracked box. Export starts at the selected time and ends at that segment's
   boundary. Processing runs locally;
   the MP4 export is 720 × 1280 and retains the source audio.

The launcher keeps the Python 3.7 product environment separate from the
optional Python 3.10 YOLO worker environment. Install Python 3.10, then create
that environment and install the optional dependency:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv-yolo\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" --index-url https://download.pytorch.org/whl/cu128
$env:BASKETBALL_YOLO_PYTHON = (Resolve-Path ".\.venv-yolo\Scripts\python.exe").Path
.\venv37\Scripts\python.exe basketball_web.py
```

The launcher verifies both environments, selects the available YOLO
interpreter, and reports whether it will use the CUDA GPU or CPU. If the
optional environment is missing or fails its check, the app starts with
optical-flow tracking and prints a warning. The YOLO option becomes available
in the page when its environment check passes. The CUDA command
installs a 3 GB GPU-enabled PyTorch package. Choose the YOLO mode, analyze the
current frame, select a detected player, then start ByteTrack. On first
inference Ultralytics may download the
`yolov8n.pt` weights; set `BASKETBALL_YOLO_MODEL` to a local weights path to use
another model. The detector uses the COCO `person` class and does not identify
the ball holder. Ultralytics licensing terms apply; review them before
redistributing or commercially deploying this optional backend.

The hybrid recommendation uses OpenCV's general-purpose HOG person detector
and a weak orange-color cue; it is not a basketball-trained model and does not
report possession probabilities. It cannot reliably identify the ball holder
or switch players after a pass, so inspect and confirm the selected box before
export. Shot boundaries are approximate, not professional broadcast-cut
detection. Review the selected segment before export; process different segments
separately, confirming the player again for each one. In the default mode, the
exported crop tracks the confirmed box using optical flow; in YOLO mode, export
reuses the ByteTrack trajectory from the preview. Both modes smooth horizontal
camera movement.
If optical-flow features are lost for at least half a second, the preview and
export display a tracking warning; choose a clearer starting frame and reselect
the player rather than trusting that portion of the crop. The feature-loss
warning is not a person-identity confidence score.

The uploaded video and temporary export are stored in the operating system's
temporary directory and are not automatically removed when the app stops.
Remove the `basketball_vertical_studio` temporary folder when you want to clear
those files.

## Reproducing the SmartVidCrop Paper

The repository contains the original SmartVidCrop implementation, pretrained
models, six RetargetVid annotators, and published crop-window results. To re-run
the published full-dataset IoU evaluation without overwriting the existing
`eval_current.txt`, run the evaluator from a temporary working directory:

```powershell
$work = Join-Path $env:TEMP "retargetvid_reproduction_eval"
New-Item -ItemType Directory -Force -Path $work | Out-Null
Set-Location $work
& "E:\py_cpp\RetargetVid-main\RetargetVid-main\venv37\Scripts\python.exe" `
  "E:\py_cpp\RetargetVid-main\RetargetVid-main\retargetvid_eval.py" `
  "E:\py_cpp\RetargetVid-main\RetargetVid-main\results"
```

To freshly run the paper's default ICIP settings on a video and evaluate both
target ratios against its six human annotations:

```powershell
Set-Location "E:\py_cpp\RetargetVid-main\RetargetVid-main"
& ".\venv37\Scripts\python.exe" reproduce_smartvidcrop.py --video-id 001
```

The script saves per-frame crop windows and an IoU/parameter summary under
`reproduction\smartvidcrop_001`. To select another supplied video, replace
`001` with its three-digit filename stem. The original paper's full 200-video
results are evaluated from the bundled `results\smartvidcrop` files; rerunning
the method from scratch on the full dataset is a separate, substantially more
time-consuming operation.


## Annotator Software
To assist the annotators in their task we implemented a graphical user interface tool which facilitates the navigation throughout the video, allows the user to set a crop window for each frame through simple drag-and-drop mouse operations, and overlays the crop window on the video frames to allow for the quick inspection of the user's decisions.

The software was implemented in C# and the source code is available in the *annotator_software* directory of this repository. You will need Visual Studio 2019 to compile the source code.



## Prerequisites
To run our SmartVidCrop method you will need Python 3. Clone this repository and make sure to download the TransNet pretrained models files (*transnet_model-F16_L3_S2_D256.data-00000-of-00001*, *transnet_model-F16_L3_S2_D256.index* and *transnet_model-F16_L3_S2_D256.meta*) from the [TransNet](https://github.com/soCzech/TransNet/tree/master/model) repository and place them in *RetargetVid/3rd_party_libs/transnetv1/* directory.
You must also have the following packages which can be simply installed via pip (in parenthesis you can find the recommended version to install):

* TensorFlow (tensorflow-gpu==1.14)
* PyTorch (torch==1.7.1+cu110)
* hdbscan (hdbscan==0.8.26)
* ffmpeg (ffmpeg==1.4)
* OpenCV (opencv-python==4.2.0)
* SciPy (scipy==1.5.1)
* ImUtils (imutils==0.5.4)
* Scikit-learn (scikit-learn==0.24.1)

SmartVidCrop also uses the following libraries
* [Unisal](https://github.com/rdroste/unisal)
* [TransNet](https://github.com/soCzech/TransNet)

pipbut these packages are already included in the 3rd_party_libs directory.

To run the evaluator software you will need Python 3.

To run the annotator software you will need .ΝΕΤ framework 3.1 runtimes for your version of Windows (you can download this from [here](https://dotnet.microsoft.com/download)).

## Citations

If you use any of this repository contents, please cite the following works:
```
@inproceedings{kapost2021afast,
title={A fast smart-cropping method and dataset for video retargeting},
author={Apostolidis, Konstantinos and Mezaris, Vasileios},
booktitle={Proc. 2021 IEEE International Conference on Image Processing (ICIP)},
year={2021},
organization={IEEE}
}

@inproceedings{kapost2021webservice,
title={A Web Service for Video Smart-Cropping},
author={Apostolidis, Konstantinos and Mezaris, Vasileios},
booktitle={Proc. 2021 IEEE Int. Symposium on Multimedia (ISM)},
year={2021},
organization={IEEE}
}
```

The full text of our above-referenced ICIP paper can be found here: [https://doi.org/10.1109/ICIP42928.2021.9506390](https://doi.org/10.1109/ICIP42928.2021.9506390)

The original videos are taken from the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset, which was introduced in the following work:

```
@article{wang2019revisiting,
title={Revisiting video saliency prediction in the deep learning era},
author={Wang, Wenguan and Shen, Jianbing and Xie, Jianwen and Cheng, Ming-Ming and Ling, Haibin and Borji, Ali},
journal={IEEE transactions on pattern analysis and machine intelligence},
volume={43},
number={1},
pages={220--237},
year={2019},
publisher={IEEE}
}
```

---

## 简体中文

### 简介

本仓库包含 **RetargetVid** 视频重定向数据集，以及 **SmartVidCrop** 快速视频裁剪方法，用于改变视频的宽高比。

我们发现，视频重定向研究通常使用各自挑选的视频进行测试，这些视频往往不公开，评估也常依赖对部分帧的人工观察。为此，我们构建并发布了 RetargetVid：一个公开的视频裁剪基准数据集，由 6 位标注者完成标注。

我们还发布了一种简单、快速且效果良好的视频裁剪方法。该方法通过新增的“过滤后聚类”处理步骤，从视频中多个可能的显著区域中选出主要关注区域。

### RetargetVid 数据集

我们从公开的 [DHF1k](https://github.com/wenguanwang/DHF1K) 数据集中选取了 200 个视频：训练集前 100 个视频（编号 001–100）和验证集中的 100 个视频（编号 601–700）。所有视频均为 16:9 宽高比，且大多数只包含一个镜头。

我们邀请了 6 位标注者，为每一帧选出适合裁剪保留的区域。具体来说，每个视频都标注两种裁剪结果，目标宽高比分别为 1:3 和 3:1。尽管实际应用中通常不会采用如此极端的宽高比，我们仍使用它们来研究高难度裁剪条件下的人类偏好。对于不那么极端的目标宽高比（例如 9:16），也可以通过测量裁剪窗口对 1:3 人工标注窗口的覆盖程度来进行评估。

每个视频的裁剪窗口标注以文本文件保存，第 *i* 行包含第 *i* 帧裁剪窗口左上角和右下角的坐标。坐标从零开始，即帧左上角为 (0,0)。标注文件共 2400 个（200 个视频 × 2 种目标宽高比 × 6 位标注者），命名格式为 *$video_id$-$target_aspect_ratio$.txt*，其中目标宽高比为 "1-3" 或 "3-1"。标注文件位于仓库的 *annotations* 目录中，每位标注者对应一个压缩文件。DHF1k 视频可通过原作者 GitHub 仓库中的[下载链接](https://github.com/wenguanwang/DHF1K)获取。

### SmartVidCrop 方法

当视频宽高比转换需要尽量减少语义失真时，我们认为裁剪方法更为合适。因此，我们提出了一种简单、快速且效果良好的视频裁剪方法，通过“过滤后聚类”步骤，从多个可能的显著区域中选出主要关注区域。该方法利用视觉显著性定位受关注区域，再通过过滤与聚类选择主要焦点。详情请参阅“引用”部分的第一篇文献。

该方法使用 Python 3 实现，主脚本为仓库中的 *smartVidCrop.py*。你可以在自己的 Python 脚本中导入该模块，也可以直接运行；运行后将调用主方法并复现论文结果。

我们还实现了裁剪质量评估机制：如果视频内容导致任何裁剪窗口都只能覆盖少量显著信息，系统可以改用填充方式。这是因为裁剪后的视频并不总能保留原视频中的所有关注区域。该功能默认关闭。若要启用此功能或调整其他参数，请参阅 *smartVidCrop.py* 中的 `sc_init_crop_params` 函数。

### 评估与更新版 SmartVidCrop 方法

若要根据 **RetargetVid** 数据集的真实标注评估你的方法，请先下载本仓库，然后在 *results* 目录中新建子目录，并为数据集中的 200 个视频分别创建一个文本文件。文件命名格式为 *$video_id$-$target_aspect_ratio$.txt*，其中目标宽高比为 "1-3" 或 "3-1"。SmartVidCrop 本身支持任意目标宽高比，但当前评估脚本仅支持 "1:3" 和 "3:1"。每行应包含裁剪窗口的（上、左、下、右）坐标。最后运行 Python 脚本 *retargetvid_eval.py*。脚本会显示 *results* 目录中各子目录的评估结果，并提示处理错误或标注不完整的情况。

评估结果是 *results* 子目录中的裁剪窗口与 RetargetVid 真实标注之间的平均相似度，以交并比（IoU）计算。

仓库的 *results* 目录包含两个子目录：
* *smartvidcrop*：本方法的结果；
* *autoflip*：Google [AutoFlip](https://google.github.io/mediapipe/solutions/autoflip) 方法的结果。

你可以使用这些结果快速复现论文评估（参见“引用”部分的第一篇文献）。评估软件使用 Python 3 实现，源代码位于仓库中的 *retargetvid_eval.py*。

在 IEEE ICIP 2021 论文发表后，我们对算法做了四项改进，并在 IEEE ISM 2021 论文中介绍。改进包括：对输入过滤聚类步骤的显著性图进行空间下采样以加快处理；增加“焦点稳定性”机制以抑制焦点突变；用 Savitzky–Golay 滤波器替代 ICIP 2021 论文中的 LOESS 时间序列平滑方法，使裁剪窗口运动更加平滑；以及采用优化后的参数。若要启用这些改进，请调用 *sc_init_crop_params* 并将可选参数 *use_best_settings* 设为 `True`。*smartVidCrop.py* 的默认设置用于复现 ICIP 论文结果。下表给出 RetargetVid 全部 6 位标注者上的平均 IoU（越高越好），以及测试电脑处理时间与视频时长之比 *t*（越低越好；低于 100% 表示快于实时处理）。

| 参数设置 | 1:3 平均 IoU | 1:3 时间比 *t* | 3:1 平均 IoU | 3:1 时间比 *t* |
| --- | ---: | ---: | ---: | ---: |
| ICIP 2021 论文设置（默认） | 49.9% | 19% | 71.4% | 20% |
| ISM 2021 论文设置（将可选参数 *use_best_settings* 设为 True） | 52.9% | 13% | 75.3% | 14% |

### 篮球视频本地裁剪（命令行）

暂时不需要网页时，可以直接处理本地视频文件。使用已安装产品依赖的 Python 环境运行：

```powershell
python .\basketball_video_cropper\crop_basketball_video.py
```

不带参数运行时，会先弹出视频选择和输出位置对话框，然后显示视频帧供你用鼠标框选球员；按 Enter/Space 确认。也可以从命令行传入视频路径及时间范围：

```powershell
python .\basketball_video_cropper\crop_basketball_video.py "D:\videos\game.mp4" --start 12 --end 28 --output "D:\results\clip.mp4"
```

可以通过 `--tracker yolo` 或 `--tracker flow` 指定 YOLOv8 + ByteTrack 或原光流跟踪；双击对应的独立程序时会自动使用对应算法。

若尚未安装产品依赖，请在所用 Python 环境中执行：

```powershell
python -m pip install -r requirements-product.txt
```

篮球裁剪器保留了两种独立程序，均为无需单独安装 Python 的 Windows 单文件程序：

```powershell
py -3.10 -m pip install -r .\basketball_video_cropper\requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\basketball_video_cropper\build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\basketball_video_cropper\build-flow.ps1
```

`BasketballVideoCropper-YOLO.exe` 使用 YOLOv8 + ByteTrack 跟踪，`BasketballVideoCropper-Flow.exe` 保留原来的光流跟踪方式。两者均可直接双击或把视频文件拖到程序图标上运行，且按原视频文件名分别保存到 `basketball_video_cropper\cropped_videos\<原视频名>\`。目标文件名已存在时会自动追加 `_2`、`_3` 等编号，不会覆盖已有视频。

YOLO 版包含 CUDA PyTorch、YOLOv8、ByteTrack、模型和 FFmpeg。构建需要先准备 `.venv-yolo`，按前文安装 `requirements-yolo.txt` 和支持 CUDA 的 PyTorch，再安装打包依赖并运行 `build.ps1`。完整 GPU 版本体积可能超过 5 GiB，首次启动会解包较长时间，运行机器还需要兼容的 NVIDIA 驱动。光流版体积较小，使用 `build-flow.ps1` 单独构建，无需 YOLO 环境。YOLO 模型只检测 COCO `person` 类，不识别篮球或判断谁持球。


### 篮球竖屏视频工作台

本仓库还包含一个可在本地浏览器运行的 MVP，用于将横屏篮球视频转换为竖屏，并尽量让所选球员保持在裁剪画面中：

1. 在仓库的 Python 环境中安装产品依赖：

```powershell
.\venv37\Scripts\python.exe -m pip install -r requirements-product.txt
```

2. 启动本地应用并检查可选 YOLO/CUDA 环境：

```powershell
.\venv37\Scripts\python.exe .\start_basketball.py
   ```

3. 打开 <http://127.0.0.1:8765> 并上传横屏视频。应用会根据颜色直方图变化估算硬切镜头，并将视频划分为多个片段。选择一个片段，在其中挑选清晰画面，点击 **Analyze current frame（分析当前帧）** 查看按排序排列的球员候选。点击候选可确认目标，也可以拖动鼠标手动框选以覆盖建议。然后点击 **Lock target and generate tracking preview（锁定目标并生成跟踪预览）**，拖动时间轴检查跟踪框。导出视频从选定时间开始，到该片段边界结束。处理在本地运行；导出 MP4 分辨率为 720 × 1280，并保留原始音轨。

启动脚本会让 Python 3.7 产品环境与可选的 Python 3.10 YOLO worker 环境保持隔离。请先安装 Python 3.10，再创建独立环境并安装可选依赖：

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv-yolo\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" --index-url https://download.pytorch.org/whl/cu128
$env:BASKETBALL_YOLO_PYTHON = (Resolve-Path ".\.venv-yolo\Scripts\python.exe").Path
.\venv37\Scripts\python.exe basketball_web.py
```

启动脚本会检查两套环境、选择可用的 YOLO 解释器，并报告将使用 CUDA GPU 还是 CPU。若可选环境不存在或检查失败，应用会显示警告并以光流跟踪模式启动。YOLO 环境通过检查后，网页才会启用 YOLO 选项。CUDA 安装命令会下载约 3 GB 的 GPU 版 PyTorch。选择 YOLO 模式、分析当前帧、选中检测到的球员，再启动 ByteTrack。首次推理时 Ultralytics 可能会下载 `yolov8n.pt` 权重；如需使用其他模型，可将 `BASKETBALL_YOLO_MODEL` 设置为本地权重文件路径。检测器使用 COCO 的 `person` 类，并不能识别持球人。重新分发或商业部署此可选后端前，请先确认 Ultralytics 的许可条款。

默认候选推荐结合了 OpenCV 通用 HOG 人体检测器和较弱的橙色区域线索；它不是篮球专用训练模型，也不提供球权概率。YOLOv8 模式检测的是通用 `person` 类，ByteTrack 负责在镜头片段内关联跟踪 ID；它们都不能可靠识别持球人，也不会在传球后自动切换到新的持球球员，因此导出前请检查并确认目标。镜头边界是估算结果，并非专业的转播切镜检测。请在导出前检查选定片段；不同片段需要分别处理，并在每段重新确认球员。默认模式使用光流跟踪，YOLO 模式的导出复用预览所生成的 ByteTrack 轨迹；两种模式均用基于 Kalman 运动模型的裁剪中心平滑来降低画面抖动。如果跟踪特征不足或 ByteTrack 连续至少半秒未检测到所选 ID，预览和导出会显示跟踪警告；遇到这种情况，请重新选择清晰帧并检查跟踪结果。检测置信度和特征状态都不是球员身份置信度。

上传的视频和临时导出文件保存在操作系统的临时目录中，应用关闭时不会自动删除。如需清理这些文件，请删除 `basketball_vertical_studio` 临时目录。

### 复现 SmartVidCrop 论文

仓库包含原始 SmartVidCrop 实现、预训练模型、RetargetVid 的 6 组标注，以及已发布的裁剪窗口结果。若要重新运行完整数据集的 IoU 评估，同时避免覆盖现有的 `eval_current.txt`，请从临时工作目录运行评估器：

```powershell
$work = Join-Path $env:TEMP "retargetvid_reproduction_eval"
New-Item -ItemType Directory -Force -Path $work | Out-Null
Set-Location $work
& "E:\py_cpp\RetargetVid-main\RetargetVid-main\venv37\Scripts\python.exe" `
  "E:\py_cpp\RetargetVid-main\RetargetVid-main\retargetvid_eval.py" `
  "E:\py_cpp\RetargetVid-main\RetargetVid-main\results"
```

若要使用论文默认的 ICIP 设置重新处理单个视频，并针对 6 位标注者的结果评估两种目标宽高比，请运行：

```powershell
Set-Location "E:\py_cpp\RetargetVid-main\RetargetVid-main"
& ".\venv37\Scripts\python.exe" reproduce_smartvidcrop.py --video-id 001
```

脚本会在 `reproduction\smartvidcrop_001` 下保存逐帧裁剪窗口以及 IoU/参数摘要。若要处理其他已提供的视频，请将 `001` 替换为对应的三位数文件名前缀。原论文在完整 200 个视频上的结果通过仓库自带的 `results\smartvidcrop` 文件评估；从头对完整数据集重新运行算法是另一项耗时显著更长的操作。

### 标注软件

为协助标注者完成任务，我们实现了一个图形界面工具，可便捷地浏览视频、通过拖放鼠标为每一帧设置裁剪窗口，并将窗口叠加显示在视频帧上，以便快速检查标注结果。

该软件使用 C# 实现，源代码位于仓库的 *annotator_software* 目录中。编译需要 Visual Studio 2019。

### 环境要求

运行 SmartVidCrop 需要 Python 3。克隆本仓库后，请从 [TransNet](https://github.com/soCzech/TransNet/tree/master/model) 仓库下载预训练模型文件（*transnet_model-F16_L3_S2_D256.data-00000-of-00001*、*transnet_model-F16_L3_S2_D256.index* 和 *transnet_model-F16_L3_S2_D256.meta*），并放入 *RetargetVid/3rd_party_libs/transnetv1/* 目录。

此外还需安装以下 Python 包，可通过 pip 安装（括号中为建议版本）：

* TensorFlow (tensorflow-gpu==1.14)
* PyTorch (torch==1.7.1+cu110)
* hdbscan (hdbscan==0.8.26)
* ffmpeg (ffmpeg==1.4)
* OpenCV (opencv-python==4.2.0)
* SciPy (scipy==1.5.1)
* ImUtils (imutils==0.5.4)
* Scikit-learn (scikit-learn==0.24.1)

SmartVidCrop 还使用以下库；这些库已包含在 *3rd_party_libs* 目录中：
* [Unisal](https://github.com/rdroste/unisal)
* [TransNet](https://github.com/soCzech/TransNet)

运行评估软件需要 Python 3。运行标注软件需要适用于当前 Windows 版本的 .NET Framework 3.1 运行时，可从[此处](https://dotnet.microsoft.com/download)下载。

### 引用

若使用本仓库中的任何内容，请引用以下文献：

```bibtex
@inproceedings{kapost2021afast,
title={A fast smart-cropping method and dataset for video retargeting},
author={Apostolidis, Konstantinos and Mezaris, Vasileios},
booktitle={Proc. 2021 IEEE International Conference on Image Processing (ICIP)},
year={2021},
organization={IEEE}
}

@inproceedings{kapost2021webservice,
title={A Web Service for Video Smart-Cropping},
author={Apostolidis, Konstantinos and Mezaris, Vasileios},
booktitle={Proc. 2021 IEEE Int. Symposium on Multimedia (ISM)},
year={2021},
organization={IEEE}
}
```

上述 ICIP 论文全文可通过此链接获取：[https://doi.org/10.1109/ICIP42928.2021.9506390](https://doi.org/10.1109/ICIP42928.2021.9506390)。

本项目使用的原始视频来自 [DHF1k](https://github.com/wenguanwang/DHF1K) 数据集，该数据集由以下论文提出：

```bibtex
@article{wang2019revisiting,
title={Revisiting video saliency prediction in the deep learning era},
author={Wang, Wenguan and Shen, Jianbing and Xie, Jianwen and Cheng, Ming-Ming and Ling, Haibin and Borji, Ali},
journal={IEEE transactions on pattern analysis and machine intelligence},
volume={43},
number={1},
pages={220--237},
year={2019},
publisher={IEEE}
}
```
