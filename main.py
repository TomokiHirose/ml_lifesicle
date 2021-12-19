import hydra
from hydra.utils import get_original_cwd
import mlflow
from omegaconf import DictConfig
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

@hydra.main(config_path=".", config_name='config')
def run(cfg: DictConfig):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    mlflow.set_tracking_uri("file://" + get_original_cwd() + "/mlruns")
    mlflow.set_experiment(cfg.nlp.experiment)

    with mlflow.start_run() as mlrun:

        mlflow.log_param("hidden_layer", cfg.nlp.hidden_layer_sizes)
        mlflow.log_param("activation", cfg.nlp.activation)

        clf = MLPClassifier(hidden_layer_sizes=[cfg.nlp.hidden_layer_sizes, cfg.nlp.hidden_layer_sizes],
                        activation=cfg.nlp.activation,
                        solver=cfg.nlp.solver,
                        batch_size=cfg.nlp.batch_size,
                        max_iter=cfg.nlp.max_iter,
                        early_stopping=cfg.nlp.early_stopping
                        ).fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        mlflow.log_metric("accracy", acc)
        mlflow.sklearn.log_model(clf, "model")

    return acc

if __name__ == '__main__':
    # python ./main.py -m "nlp.hidden_layer_sizes=choice(16,32,64,128,256)" "nlp.activation=choice(logistic, tanh, relu)
    run()
