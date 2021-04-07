import pickle
from sklearn_porter import Porter

SVC = pickle.load(open("BenchmarkModels/LinearSVC_bm_model", "rb"))

# print(SVC)

SVC_porter = Porter(SVC, language='js')

SVC_js = SVC_porter.export(embed_data=True)
with open('JSPorts/LinearSVC.js', 'w') as f:
    f.write(SVC_js)
