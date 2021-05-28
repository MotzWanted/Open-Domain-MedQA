# Docker Services
ElasticSearch and Kibana runs within Docker containers
`docker compose up -d`

Following endpoints are then available:
- [ElasticSearch 7.13](localhost:9200)
- [Kibana 7.13](localhost:5601)

**Persistent storage** can be found in the .services folder. E.g. delete it to remove all data such as indexes.