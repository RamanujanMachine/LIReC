# LIReC Application on BOINC

## Configuration and Execution Overview

The LIReC application, as configured for the BOINC platform, operates primarily through JSON configuration files that specify the behavior and parameters for each computational job. Here's how the system works:

### JSON Configuration

- **Input**: A JSON file is passed as a parameter when initiating a job. The content of this file is used to configure the specifics of the computation.
- **Fallback**: If the provided JSON configuration is invalid or cannot be parsed, the system will revert to using a hardcoded configuration defined in `config.py`.

### Output and Results Handling

- **Output File**: The results of new relations discovered during the computation (up to 3 per job) are written to `output.json`. This file is then fetched by BOINC to send the results back to the server.

### File and Job Management

- **Input Files Directory**: All JSON configuration files should be placed in:
  ```
  /home/boincadm/projects/boinc/data/lirec_input
  ```

This directory contains the list of JSON files that will be used to configure the LIReC jobs.

- **Job Launching**:
- Navigate to the directory:
  ```
  /home/boincadm/projects/boinc/bin
  ```
- Execute the command:
  ```
  ./tasks_avail_workgen_lirec
  ```
  This script will launch a job for each JSON file found in the `data/lirec_input` directory and subsequently clear the folder.

### Monitoring and Retrieving Results

- **Job Tracking**: You can navigate to the following URL to monitor the jobs:
https://rnma.xyz/1b50635216bc11eca7a35f5cf61fb0ec/db_action.php?table=workunit&appid=7&detail=low

Here, the name of the result file for each job will be listed under the "XML doc out" section.

- **Results Retrieval**:
- On the BOINC server, navigate to the upload folder and use the following command to find the result file:
  ```
  find . -type f -name "lirec_wu_20240927_1541_config_60_0_r156311_0"
  ```
  Replace `"lirec_wu_20240927_1541_config_60_0_r156311_0"` with the actual name of your result file as listed on the job tracking page.

- **Result Contents**: Reading the specified result file will provide a list of results that the job discovered.

### Application Version Handling

- **Version Check**: The `tasks_avail_workgen_lirec` script checks for the latest version of the application to ensure compatibility.
- **Version Compatibility**: If the version of the application is unknown or not listed, the job will not be executed, ensuring that only compatible versions process jobs.

Ensure that all configurations and executable scripts are properly set up and that the correct permissions are in place to execute and modify files as described.
