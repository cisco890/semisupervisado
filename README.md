# Semisupervisado

https://uvggt-my.sharepoint.com/:w:/g/personal/mar23617_uvg_edu_gt/IQBojZO3crgWSJHEhDklRA7gAcolA45Rt54eqJBZkWcS1EQ?e=vk2Jpv

## Propagacion de etiquetas

Este repo incluye un ejemplo ejecutado de aprendizaje semisupervisado con Label Propagation:

- Notebook: `notebooks/label_propagation_semisupervisado.ipynb`
- Documento comparativo: `docs/label_propagation_comparativa.md`

Para reproducirlo:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_label_propagation_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/label_propagation_semisupervisado.ipynb --output label_propagation_semisupervisado.ipynb --output-dir notebooks
```
