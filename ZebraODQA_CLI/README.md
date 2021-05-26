# ZebraODQA
Utility tool

# Transferring data to the GPU cluster

1. Ensure you are connected to Compute VPN

2. write your `~/.ssh/config` file

You can replace `valv` with your student id.

```shell
Host cronus
    HostName cronus.compute.dtu.dk
    User valv

Host coeus
    HostName coeus.compute.dtu.dk
    User valv

Host hyperion
    HostName hyperion.compute.dtu.dk
    User valv

Host themis
    HostName themis.compute.dtu.dk
    User valv

Host theia
    HostName theia.compute.dtu.dk
    User valv

Host tethys
    HostName tethys.compute.dtu.dk
    User valv

Host mnemosyne
    HostName mnemosyne.compute.dtu.dk
    User valv

Host oceanus
    HostName oceanus.compute.dtu.dk
    User valv

Host rhea
    HostName rhea.compute.dtu.dk
    User valv

Host phoebe
    HostName phoebe.compute.dtu.dk
    User valv

Host crius 
    HostName crius.compute.dtu.dk
    User valv

Host iapetus
    HostName iapetus.compute.dtu.dk
    User valv
```

3. use `scp` or `rsync` to transfer files or directories

for files, directories
```shell  
rsync -avh <source_dir/file> <hostname>:/scratch/<studentID>
```
or (for files)

```shell
scp <file> <hostname>:/scratch/<studentID>  
```

if you skipped step 2, replace `<hostname>` with `<studentID>@<cluster>.compute.dtu.dk`

4. Optional: password-free login
  
  Alternatively, you can register an ssh key on the cluster, so you don't have to use your own password. You can follow this [tutorial](http://www.linuxproblem.org/art_9.html).
