# fit-BEL
End-to-end code to estimate AGN properties from a single spectrum by fitting the continuum and a Broad Emission Line (BEL)

## Usage

Clone the repository on your local machine:
```bash
git clone https://github.com/Alexperiments/fit-BEL.git
```

Enter the newly created folder:
```bash
cd fit-BEL
```

Create a virtual environment
```bash
python -m venv my_env
```

Activate the virtual environment for Unix OS:
```bash
source my_env/bin/activate
```

If you are running Windows use this instead:
```bash
.\my_venv\Scripts\activate.ps1
```

Install required dependencies into the virtual environment:
```bash
pip install -r requirements.txt
```

Run fit-BEL as:
```bash
python fit-BEL.py examples/sample.fits -z 3.1 -e 2.1
```

Run:
```bash
python fit-BEL.py -h
```
to get more information about the optional arguments

## Development

### Tasks
- [ ] Adapt the [original code](https://github.com/AleD1996/diana_et_al_2021) to a single script: 
    - [x] Basic spectrum plot
    - [x] Add interval selection procedure
    - [x] Add continuum fit procedure
    - [x] Add mask selection procedure
    - [x] Add fit selection procedure
    - [x] Add parameters estimation
    - [x] Refactor the code to allow models modularity
    - [ ] Add error estimate procedure
    
### Bugs
- [x] Fix the zoom reset
- [x] Fix the masks re-draw
- [x] Fix FWHM calculation

### New features
- [ ] Add different file formats in addition to `.fits` and `.txt`
- [ ] Add different fit components in addition to gaussian ones.
