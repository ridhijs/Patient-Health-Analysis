* Install python requirements

```python
python -m pip install -r requirements.txt
```
* Create a virtual environement

```python
cd [Path to datathon root folder]
python -m pip install virtualenv
python -m virtualenv dt
```

* Adding Virtual Env to Jupyter

```python
python -m ipykernel install --user --name=dt
```


* Get Kaggle Dataset

  - Get the api key
      - kaggle.com/[USERNAME]/account
      - ![image](https://user-images.githubusercontent.com/6872080/73084197-85dd9b00-3e9a-11ea-9e32-2666d356f47a.png)


  - Bash
    -
      ```bash
      cd [Path to datathon root folder]
      python -m Datathon.Utils.getKaggleDataset
      ```
