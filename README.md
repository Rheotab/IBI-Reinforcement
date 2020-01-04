# IBI-Reinforcement


### Introduction 

Ce rapport a pour but de décrire ce qui a été réalisé dans ce projet. 
Notre objectif dans un premier temps était d'aller le plus loin possible (extensions comprises).


### Préliminaires

- pyenv 
- pycharm
- python3
- pytorch

### Partie 1

- Agent Aleatoire. L'agent renvoi une action aléatoire de son action_space.

- Méthode d'évaluation de l'agent : 
    - Courbe de résultats, ou on note pour chaque episode (x) le résultats de l'agent (y).
    - Courbe d'erreur. On note l'erreur commise (y) a chaque pas d'apprentissage (x) de l'agent.
    - Courbe de Qvaleurs. On note pour chaque action (x, couleur) choisi par l'agent (or exploration) sa qvaleurs (y)  

- L'experience replay a été fait avec une simple liste python. On aurait pu utiliser une dequeu mais on trouvé plus
 simple de procéder comme on l'a fait. 
 
- Les mini-batch sont crées à partir de la mémoire, grâce au paquet random.  

- Le réseau a été fait en utilisant les modules pytorch.nn. On a fait plusieurs tests avec une, deux, ou trois couches de neuronnes cachées. 

- Les deux politiques (e-greedy / exploration de Boltzmann) ont été faites. 
On a choisi d'utiliser e-greedy puisqu'on est plus familier avec et que la politique importe peu, tant qu'il y a une exploration.  
Epsilon va se degrader de plus en plus. Pour finalement atteindre une valeur minimale.

- L'erreur (équation de Bellman) est calculé grâce au module mse_loss qui fait le calcul à notre place. 

- La mise à jour du target network a été faite grâce au compteur, plutôt que progressivement.

- Résultats : Notre agent apprend correctement mais manque de stabilité. On aurait aimé avoir une courbe de progression linéaire mais ce n'est pas le cas.  
Vous trouverez une vidéo de cartpole qui fait le meilleur score (500).

### Partie 2 

- Le preprocessing du breakout a été fait. Lorsque l'agent fait une action on a: 
    - fait 4 fois l'action sur l'environnement en gardant toute les observations/rewards.
    - Pour chaque frames et pour chaque pixel, on a pris le maximum entre la frame courante et la frame d'avant (mise en mémoire de la derniere frame pour l'action suivante)
    - Si nous sommes en mode apprentissage. L'agent perd la partie lorsqu'il perd une vie. 
    - Si la fin de la partie arrive avant les 4 répetitions. On copie simplement la derniere frame le nombre de fois restantes.
    - On prend le maximum des trois couleurs de chaque pixels (réduction de dimensions 3 -> 1)
    - On reshape en utilisant OpenCV
    - Il est possible de normaliser l'image
    - Lorsqu'on reset l'environnement. On produit 4 fois l'operation NO-OP pour renvoyer 4 frames à l'agent.

- Puisqu'on est passé par une liste python, le buffer d'exeperience n'a pas eu besoin d'être adapté.

- On a fait deux versions différentes du réseaux convolutionnels. Une qui correspond à l'architecture décrite dans le papier publié dans Nature, l'autre plus "custom".
- On initialise la mémoire de l'agent (40 000 episodes), pour eviter de sur-apprendre les premieres configurations. 
- Résultats : L'apprentissage ne marche pas.... :'(