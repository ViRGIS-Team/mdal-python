cp ../LICENCE .
cp ../README.rst .
python -m build
python3 -m twine upload --repository testpypi dist/*