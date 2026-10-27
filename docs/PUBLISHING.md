# Pubblicazione del repository

La cartella contiene una copia separata e selezionata del progetto. Nessun file
del workspace originale deve essere aggiunto al repository pubblico.
Dataset derivato e predizioni sono inclusi su richiesta del proprietario;
log grezzi, DAT, FEM e modelli serializzati sono esclusi.

## 1. Verifiche prima della pubblicazione

- Confermare attribuzione degli autori, autorizzazioni istituzionali e licenza
  appropriata per codice e dati. `RIGHTS.md` non e' una licenza open-source.
- Controllare README, dati derivati e figure. Il nome definitivo del sistema
  e' AIMS; il repository e' `alessandrocabras/aims-thesis`.
- Non copiare credenziali, log, dati riservati o modelli nella copia pubblica.

Aprire un terminale nella cartella `release/aims-thesis`, quindi:

```powershell
python scripts/check_release.py
python -m unittest discover -s tests -v
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
git commit -m "Prepare thesis review repository"
```

Il repository locale e' gia' inizializzato sul branch `main`. Non e' stato
creato un commit con un'identita' presunta. Se Git richiede l'identita', usare
`git config user.name "Nome Cognome"` e `git config user.email "EMAIL_SCELTA"`
in questa cartella; si puo' scegliere l'indirizzo noreply fornito da GitHub.

## 2. Pubblicare sul repository GitHub creato

Il repository e' https://github.com/alessandrocabras/aims-thesis.
Il remoto `origin` e' gia' configurato localmente. Verificarlo e poi eseguire:

```powershell
git remote -v
git push -u origin main
```

Per una nuova copia locale non ancora collegata al remoto:

```powershell
git remote add origin https://github.com/alessandrocabras/aims-thesis.git
```

Il push non e' stato eseguito automaticamente. Se il remoto contiene gia'
commit (per esempio un README creato dal sito), non usare `--force`: integrare
prima la storia remota. Istruzioni ufficiali:
[importazione di codice locale](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
e [gh repo create](https://cli.github.com/manual/gh_repo_create).

## 3. Versione per i revisori

Dopo aver controllato la pagina pubblica e i controlli automatici, creare un
tag identificabile, ad esempio:

```powershell
git tag -a v0.1.0-review -m "Thesis review snapshot"
git push origin v0.1.0-review
```

Condividere il link al tag oltre al link al repository, cosi' i revisori
possono identificare la versione esatta. Conservare una copia privata del
workspace completo: questo repository non e' un suo backup integrale.
