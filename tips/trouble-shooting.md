### Azure [Blobfuse](https://github.com/Azure/azure-storage-fuse) mount failure
  * Error: *error while loading hsared libraries: libgnutls.so.26: cannot open shared object file: No such file or directory*
  * Context: already installed the blobfuse package, however it fails when trying to mount the blob to local machine via `./blobfuse {mount local target folder path/name} --tmp-path=/mnt/blobfusetmp -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 --config-file=../connection.cfg`.
  * Reason: the installed blobfuse version does not match the current OS version. For example, you install the version for Ubuntu 16, but the current OS is Ubuntu 18.
  * Solution: uninstall the current mismatch version and reinstall the correct version. There is a great answer [here](https://github.com/Azure/azure-storage-fuse/issues/247).
    > 1. wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
    > 2. sudo dpkg -i packages-microsoft-prod.deb
    > 3. sudo apt-get update
    > 4. sudo apt-get install blobfuse

### Azure ML Computing experiment (GPU/CPU clusters) submit error
 * Error: *ValueError: ZIP does not support timestamps before 1980*
 * Context: when submitting the training to AML, it fails with the error as above.
 * Reason: there are some hidden folder/files under the source code folder (i.e. the location you specify the code to submit and run) has bad timestamp which is before 1980. The reason of why the timestamp wrong, it might be the file/foldered created by the tool or in jupyter and lead to the bad timestamp (in our case the time stamp is 1969).
 * Solution: go and delete the hidden folder/files with bad timestamp, or change the timestamp.
