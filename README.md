# talks-in-maths

```
git clone “link”

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

touch README.md
touch .gitignore
touch requirements.txt
touch pyproject.toml
touch artifacts/.gitkeep

```

add `.venv` to `.gitignore`  

test if homebrew is installed

`which brew`

if not:

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

depois pacotes basicos: 

`brew install cairo pango pkg-config`

instalando LaTeX:

`curl -sL "https://tinytex.yihui.org/install-bin-unix.sh" | sh`

teste:

```
which latex
which pdflatex
which dvisvgm
```

instale Manim no venv:
```
source .venv/bin/activate
pip install manim
```


