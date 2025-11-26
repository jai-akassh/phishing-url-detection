import re
from urllib.parse import urlparse
import tldextract

# Words commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    'login', 'secure', 'update', 'free', 'account',
    'banking', 'verify', 'confirm', 'urgent', 'alert'
]

def has_ip(url: str) -> int:
    """Check if URL contains an IP address."""
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    return int(bool(re.search(ip_pattern, url)))

def extract_features(url: str) -> dict:
    """Extract numerical features from a URL."""
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)

    features = {}

    # URL structure features
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['has_at'] = int('@' in url)
    features['has_ip'] = has_ip(url)
    features['is_https'] = int(parsed.scheme == 'https')

    # Domain features
    features['domain_length'] = len(domain_info.domain)
    features['subdomain_length'] = len(domain_info.subdomain)
    features['num_subdomains'] = (
        domain_info.subdomain.count('.') + 1
        if domain_info.subdomain else 0
    )

    # Suspicious keyword count
    lower_url = url.lower()
    features['num_suspicious_words'] = sum(
        word in lower_url for word in SUSPICIOUS_WORDS
    )

    return features

# Quick test
if __name__ == "__main__":
    print(extract_features("http://secure-login-paypal.com/verify/account"))
