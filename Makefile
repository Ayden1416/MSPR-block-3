# Variables de configuration
PYTHON := python
PIP := pip
VENV_NAME := venv
DATA_DIR := data
RESULTS_DIR := results
DB_FILE := data/dataset.sqlite

# Détection de l'OS pour des commandes spécifiques à la plateforme
ifeq ($(OS),Windows_NT)
	RM := rmdir /s /q
	MKDIR := mkdir
	VENV_ACTIVATE := $(VENV_NAME)\Scripts\activate
	SEP := \\
else
	RM := rm -rf
	MKDIR := mkdir -p
	VENV_ACTIVATE := . $(VENV_NAME)/bin/activate
	SEP := /
endif

# Cibles principales
.PHONY: all clean setup run_pipeline run_generation run_learning help

all: setup run_pipeline

# Installation de l'environnement
setup: create_venv install_deps create_dirs

create_venv:
	@echo "Création de l'environnement virtuel..."
	$(PYTHON) -m venv $(VENV_NAME)

install_deps: create_venv
	@echo "Installation des dépendances..."
ifeq ($(OS),Windows_NT)
	$(VENV_NAME)\Scripts\pip install -r requirements.txt
else
	$(VENV_NAME)/bin/pip install -r requirements.txt
endif

create_dirs:
	@echo "Création des répertoires nécessaires..."
ifeq ($(OS),Windows_NT)
	-if not exist "$(DATA_DIR)" $(MKDIR) "$(DATA_DIR)"
	-if not exist "$(RESULTS_DIR)" $(MKDIR) "$(RESULTS_DIR)"
	-if not exist "$(RESULTS_DIR)\plots" $(MKDIR) "$(RESULTS_DIR)\plots"
	-if not exist "$(RESULTS_DIR)\reports" $(MKDIR) "$(RESULTS_DIR)\reports"
else
	$(MKDIR) $(DATA_DIR)
	$(MKDIR) $(RESULTS_DIR)
	$(MKDIR) $(RESULTS_DIR)/plots
	$(MKDIR) $(RESULTS_DIR)/reports
endif

# Exécution de la pipeline
run_pipeline: run_generation run_learning

# Génération du jeu de données
run_generation:
	@echo "Génération du jeu de données..."
ifeq ($(OS),Windows_NT)
	$(VENV_NAME)\Scripts\python generate_dataset.py
else
	$(VENV_NAME)/bin/python generate_dataset.py
endif

# Exécution de l'apprentissage et des prédictions
run_learning: $(DB_FILE)
	@echo "Exécution des modèles d'apprentissage et des prédictions..."
ifeq ($(OS),Windows_NT)
	$(VENV_NAME)\Scripts\python learning.py
else
	$(VENV_NAME)/bin/python learning.py
endif

# Nettoyage des fichiers générés
clean:
	@echo "Nettoyage des fichiers générés..."
ifeq ($(OS),Windows_NT)
	-if exist "$(DB_FILE)" del "$(DB_FILE)"
	-if exist "$(RESULTS_DIR)\plots\*" del "$(RESULTS_DIR)\plots\*"
	-if exist "$(RESULTS_DIR)\reports\*" del "$(RESULTS_DIR)\reports\*"
else
	-$(RM) $(DB_FILE)
	-$(RM) $(RESULTS_DIR)/plots/*
	-$(RM) $(RESULTS_DIR)/reports/*
endif

# Nettoyage complet (y compris l'environnement virtuel)
clean_all: clean
	@echo "Suppression de l'environnement virtuel..."
	-$(RM) $(VENV_NAME)

# Vérification des dépendances
check_deps:
	@echo "Vérification des dépendances..."
ifeq ($(OS),Windows_NT)
	$(VENV_NAME)\Scripts\pip list
else
	$(VENV_NAME)/bin/pip list
endif

# Règle pour générer le fichier requirements.txt (à exécuter après avoir installé manuellement les dépendances)
generate_requirements:
	@echo "Génération du fichier requirements.txt..."
ifeq ($(OS),Windows_NT)
	$(VENV_NAME)\Scripts\pip freeze > requirements.txt
else
	$(VENV_NAME)/bin/pip freeze > requirements.txt
endif

# Aide
help:
	@echo "Makefile pour l'analyse électorale - Compatible Windows/macOS/Linux"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  make setup          - Installe l'environnement virtuel et les dépendances"
	@echo "  make run_pipeline   - Exécute toute la pipeline (génération + apprentissage)"
	@echo "  make run_generation - Génère uniquement le jeu de données"
	@echo "  make run_learning   - Exécute uniquement les modèles d'apprentissage"
	@echo "  make clean          - Nettoie les fichiers générés"
	@echo "  make clean_all      - Nettoie tout, y compris l'environnement virtuel"
	@echo "  make check_deps     - Vérifie les dépendances installées"
	@echo "  make help           - Affiche cette aide"
	@echo ""
	@echo "Pour une exécution complète, utilisez simplement: make"