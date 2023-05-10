import NetworkEnv.general_cache_allocation_algorithm as ca
import NetworkEnv.general_cache_replacement_algorithm as cr
import NetworkEnv.network as nt

def caching(network:nt):
    getattr(ca,network.cache_allocation_algorithm)(network)