
IF [%1]==[] GOTO :help

SET CWD=%cd%
SET CONDA_ROOT=%1
SET CONDA_PATH=%CONDA_ROOT%\condabin
SET CONDA_ENVS=%CONDA_ROOT%\envs
SET CONDA_EXE=%CONDA_PATH%\conda.bat
SET CONDA_ENV_NAME=py3_lpdgen
SET CONDA_ENV_PATH=%CONDA_ENVS%\%CONDA_ENV_NAME%
SET GPU_YES=%2

CALL %CONDA_EXE% create -n %CONDA_ENV_NAME% -c conda-forge python=3 numpy scipy matplotlib scikit-image --yes
CALL %CONDA_PATH%\conda.bat activate %CONDA_ENV_NAME%
COPY .condarc %CONDA_ENV_PATH%\

IF "%GPU_YES%"=="gpu" (
    CALL %CONDA_EXE% install -n %CONDA_ENV_NAME% -c pytorch pytorch torchvision torchaudio --yes
) ELSE IF "%GPU_YES%"=="cpu" (
    CALL %CONDA_EXE% install -n %CONDA_ENV_NAME% -c pytorch pytorch torchvision torchaudio cpuonly --yes
) ELSE (
    ECHO Unknown value for DEVICE_OPTION
    GOTO :help
)

CALL %CONDA_EXE% install -n %CONDA_ENV_NAME% -c conda-forge pytest pandas pep8 numba imageio --yes
CALL %CONDA_EXE% install -n %CONDA_ENV_NAME% -c conda-forge ipykernel sphinx scikit-learn --yes
CALL %CONDA_EXE% install -n %CONDA_ENV_NAME% -c conda-forge pillow opencv pytorch-lightning --yes

GOTO :end

:help
ECHO Usage: setup_environments.bat CONDA_ROOT DEVICE_OPTION
ECHO    ACTION ........... Root folder of your anaconda installation
ECHO    DEVICE_OPTION .... Device option, either "cpu" or "gpu"
GOTO :end

:end
pause

