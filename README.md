# Streamlit Sample App 
This is a Streamlit Sample App. See [Here](#vs) for a description of Streamlit and a comparism of the visualisation/dashboard frameworks that we offer as default sample apps. 

## Overview
1. [Starting Streamlit Locally](#start)
   
2. [Using Foundry Data](#foundry)
   
3. [Build Image locally](#build-local)

4. [Run Image locally](#run-image-local)
   
5. [Dash vs. Streamlit vs. RShiny ](#vs)

## Starting Streamlit Locally <a name="start"></a>

Follow these steps to create a local development server.

### Create an Environment

For local development we recommend you to use mambaforge for the creation of environments. Follow [This Guide](https://palantir.mcloud.merckgroup.com/workspace/preview-app/ri.blobster.main.pdf.15f69a1f-1a72-416e-8ffe-b1b09036a574) to install mambaforge. 

```bash
$ mamba env create -f environment.yml
$ mamba activate streamlit
```


### Start the Streamlit Server

In order to run the Streamlit server simply navigate to the src folder from root.

```bash
(streamlit)$ cd src
```

Start the server like this. See more detailed instructions in [This Guide](https://docs.streamlit.io/library/get-started)

```bash
(streamlit)$ streamlit run Welcome.py
```  

## Using Foundry Data <a name="foundry"></a>

We can use the Foundry Rest API to pull data from foundry and even write back to foundry straight from the application.  
```python
# Retrieving data with the FoundryRestClient
from foundry_dev_tools import FoundryRestClient
query = "SELECT * FROM `/Group Functions/mgf-use-case-uptimize-app-service-public-content/data/iris`"
foundry_client = FoundryRestClient()
data = foundry_client.query_foundry_sql(query=query)
```

For more code examples see the `src/pages` folder. 

## Build Image Locally <a name="build-local"></a>

Make sure to run this without VPN enabled to avoid the error `x509: certificate signed by unknown authority`: 
```bash
export DOCKER_BUILDKIT=1
docker build --pull \
             --build-arg BUILDKIT_INLINE_CACHE=1 \
             -t sample-app-streamlit:local-snapshot \
             .
```

## Run Image locally <a name="run-image-local"></a>

The following snippet runs the app on port 8501, open here: http://localhost:8501/

```bash
$ docker run --rm \
             -it \
             -p 8501:8080 \
             -e AWS_EXECUTION_ENV=AWS_ECS_FARGATE \
             -e FOUNDRY_DEV_TOOLS_JWT=$(cat ~/.foundry-dev-tools/config | grep jwt | cut -c 5-) \
             -e FOUNDRY_DEV_TOOLS_FOUNDRY_URL=$(cat ~/.foundry-dev-tools/config | grep foundry_url | cut -c 13-) \
             sample-app-streamlit:local-snapshot
```

### Run with reverse proxy (platform development only)

This setup emulates the exact setup on the app service. Install `mitmproxy` with `brew install mitmproxy`.
For app development purposes you want to use the *above* instructions!

Create a python file with the following content:

```python
import re
from os import path

with open(path.expanduser("~/.foundry-dev-tools/config"), encoding="UTF-8") as file:
    jwt = file.read()
result = re.search("jwt=(.*)\n", jwt)
foundry_token = result.group(1)


def request(flow):
    flow.request.headers["X-Foundry-AccessToken"] = foundry_token
    flow.request.headers["X-Foundry-User"] = "MUID@eu.merckgroup.com"
```

Start mitmproxy with:

```bash
mitmproxy --mode upstream:http://localhost:8501 -s <path-to-file-above.py>
```

Start docker image with:

```bash
$ docker run --rm \
             -it \
             -p 8501:8080 \
             -e AWS_EXECUTION_ENV=AWS_ECS_FARGATE \
             -e FOUNDRY_DEV_TOOLS_FOUNDRY_URL=$(cat ~/.foundry-dev-tools/config | grep foundry_url | cut -c 13-) \
             sample-app-streamlit:local-snapshot
```

Open streamlit here: http://localhost:8080/

## Dash vs. Streamlit vs. RShiny <a name="vs"></a>
So far we offer first class support for the three visualisation frameworks Dash, Streamlit and RShiny. All of these three tools offer unique capabilities to create visualizations and dashboards. Which app you choose comes down to personal preference of language support and requirements for the app that you want to build. See below for a brief categorization of the three apps.  

**Dash** offers a point & click interface written in python that exeeds most dashboard tools. It emphasizes the interactivity of widgets and input components thorugh callbacks. It is slighlty more complex then the streamlit interface but allows for more flexibility that way.  

**Streamlit** emphasizes simplicity and therefore is a great choice if you are just starting out with Pyhton dashboards or want fast prototyping. However you might run into issues if you have complicated requirements since Streamlit uses a concept of persistent state and it is generally not as costumizable.  

**RShiny** is a great option if you prefer to perform data analysis in R. It comes with an extensive library of default components, but you can even write your own widgets and Javascript actions.