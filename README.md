# LLM from scratch

## Activate env

conda activate llmfs

## Install requirements (conda or pip)

> pip install -r requirements.txt

## [Issue](https://github.com/bkerler/edl/issues/547) with pylzma library

The current clang version fails to compile this library, so downgrade temporary to 14:

> brew install llvm@14
> export PATH="/opt/homebrew/opt/llvm@14/bin:$PATH"
> pip install # pylzma
> brew uninstall llvm@14

## Pytorch

> conda install pytorch torchvision -c pytorch

## VsCode

> Ctrl + Shift + P
> Python: select interpreter

And then just select the correct environment
