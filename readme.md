Ce répertoire est fourni exclusivement aux candidats de la 1ère édition du challenge AI4IA.

Il est organisé de la manière suivante:
.
|--example_AI4IA_phase1.ipynb -> un notebook d'exemple illustrant l'attendu et présentant les différents outils mis à disposition
|--requirements.txt -> un fichier permettant l'installation des modules nécessaires à l'exécution du notebook d'exemple (pip3 install -r requirements.txt)
|--calc_metric_on_sagemaker.py -> script permettant de lancer l'évaluation des performances d'un modèle sur une instance AWS
|--data -> le répertoire contenant les datasets 
    |--DataSet_ex -> dataset d'exemple exploité dans le notebook d'exemple
    |--DataSet_phase1 -> le dataset devant être exploité par les candidats durant la phase 1
|--sources
    |--sagemaker_api.py -> script pour entrainer votre modèle localement ou sur des instances AWS (utilisable en tant que point d'entrée d'un estimateur sagemaker)
    |--calc_metrics.py  -> script permettant de calculer l'ensemble des métriques/éléments quantitatifs d'évaluation des performances de votre modèle. Peut être lancé localement ou sur une instance AWS via calc_metrics_on_sagemaker.py (utilisation en tant que point d'entrée d'un estimateur sagemaker). Ce script considère, entre autres entrées, le fichier de définition de votre modèle, les hyperparamètres choisis pour son entraînement, et un dataset de test (fichier csv);
    |--utilities
        |--model_api.py -> définition d'une classe 'virtuelle' permettant par héritage d'implémenter l'interface entre votre modèle et les outils de test et d'évaluation
        |--my_model_.py -> définition de votre modèle et de son interface avec les outils de test et d'évaluation (définition d'une class MyModel héritant nécessairement de la classe ModelApi)
        |--test_submission.py -> définition d'une classe de tests unitaires (et lancement) afin de vérifier que la définition de votre modèle est conforme à l'attendu. Il est également vivement conseillé de vérifier, avant toute soumission, que la définition du modèle permet le lancement en local ou sur des machines Amazon des scripts de calcul des métriques décrits ci-dessus
        |--utility_functions.py -> des méthodes utiles pour le chargement de données etc... pourra être enrichi
        
Les organisateurs se tiennent à votre disposition pour toute question technique ou tout problème rencontré dans l'utilisation de ces scripts: vous pouvez à tout moment transmettre un message aux organisateurs via la plateforme agorize. Le support technique tâchera de vous répondre dans les meilleurs délais.
D'autre part, des versions mises à jour des scripts pourront éventuellement vous être transmis en cours de challenge par les organisateurs.


