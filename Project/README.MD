# Lezen en laden van Jupyter notebooks

Helaas, Jupyter notebooks en versiebeheer via Git verlopen niet zo lekker. Daarom wordt ```jupytext``` gebruikt om notebooks om te zetten van en naar Python-scripts, die wel onder versiebeheer kunnen vallen.

Je kunt notebooks omzetten via het volgende commando:

```jupytext --to py [FILE_NAME].ipynb```

Na het uitvoeren van het commando wordt een nieuw bestand aangemaakt, genaamd ```[FILE_NAME].py```.

Met het volgende commando kunnen Python-scripts omgezet worden naar Jupyter notebooks. Je kunt niet elk Python-bestand vlekkeloos omzetten naar een Jupyter notebook: let vooral op de comments die in het Python-bestand staan.

````jupytext --to ipynb [FILE_NAME].py````

Na het uitvoeren van het commando wordt een nieuw bestand aangemaakt, genaamd ```[FILE_NAME].ipynb```.
