use pretty_assertions::assert_eq;

use super::*;

fn config() -> KubernetesGuestConfig {
    KubernetesGuestConfig {
        proxy_host: "kubernetes.proxy.test".into(),
        clusters: vec![
            KubernetesGuestCluster {
                name: "alpha".into(),
                contexts: vec!["ts-alpha".into()],
            },
            KubernetesGuestCluster {
                name: "123".into(),
                contexts: vec!["no".into()],
            },
        ],
        local_proxy_port: 3128,
        placeholder_token: "123".into(),
    }
}

#[test]
fn the_profile_writes_a_quoted_placeholder_kubeconfig_through_the_local_proxy() {
    let profile = config().profile().expect("profile");

    assert_eq!(profile.name, "kubernetes");
    assert_eq!(
        profile.environment,
        HashMap::from([("KUBECONFIG".into(), "/tmp/mirror-kubeconfig.yaml".into())])
    );
    assert_eq!(profile.endpoints, Vec::new());
    assert_eq!(profile.files.len(), 1);
    assert_eq!(profile.files[0].path, "/tmp/mirror-kubeconfig.yaml");
    assert_eq!(profile.files[0].mode, 0o600);
    assert_eq!(
        String::from_utf8(profile.files[0].contents.clone()).expect("kubeconfig text"),
        "apiVersion: v1\n\
         kind: Config\n\
         preferences: {}\n\
         clusters:\n\
         - name: \"alpha\"\n\
         \x20 cluster:\n\
         \x20   server: \"https://kubernetes.proxy.test/clusters/alpha\"\n\
         \x20   certificate-authority: \"/tmp/mirror-egress-ca.pem\"\n\
         \x20   proxy-url: \"http://127.0.0.1:3128\"\n\
         - name: \"123\"\n\
         \x20 cluster:\n\
         \x20   server: \"https://kubernetes.proxy.test/clusters/123\"\n\
         \x20   certificate-authority: \"/tmp/mirror-egress-ca.pem\"\n\
         \x20   proxy-url: \"http://127.0.0.1:3128\"\n\
         users:\n\
         - name: \"mirror\"\n\
         \x20 user:\n\
         \x20   token: \"123\"\n\
         contexts:\n\
         - name: \"alpha\"\n\
         \x20 context:\n\
         \x20   cluster: \"alpha\"\n\
         \x20   user: \"mirror\"\n\
         - name: \"ts-alpha\"\n\
         \x20 context:\n\
         \x20   cluster: \"alpha\"\n\
         \x20   user: \"mirror\"\n\
         - name: \"123\"\n\
         \x20 context:\n\
         \x20   cluster: \"123\"\n\
         \x20   user: \"mirror\"\n\
         - name: \"no\"\n\
         \x20 context:\n\
         \x20   cluster: \"123\"\n\
         \x20   user: \"mirror\"\n\
         current-context: \"alpha\"\n"
    );
}

#[test]
fn malformed_catalogs_are_refused() {
    let mut no_clusters = config();
    no_clusters.clusters.clear();
    let mut repeated_context = config();
    repeated_context.clusters[1].contexts = vec!["ts-alpha".into()];
    let mut odd_name = config();
    odd_name.clusters[0].name = "Alpha Cluster".into();
    let mut no_port = config();
    no_port.local_proxy_port = 0;
    let mut odd_host = config();
    odd_host.proxy_host = "kubernetes.proxy.test/clusters".into();
    let mut quoted_token = config();
    quoted_token.placeholder_token = "token: \"injected\"".into();

    assert_eq!(
        no_clusters.profile(),
        Err("the kubeconfig would list no clusters".into())
    );
    assert_eq!(
        repeated_context.profile(),
        Err("context \"ts-alpha\" is listed twice".into())
    );
    assert_eq!(
        odd_name.profile(),
        Err("cluster or context name \"Alpha Cluster\" is not a DNS label".into())
    );
    assert_eq!(
        no_port.profile(),
        Err("the sandbox local proxy port is unset".into())
    );
    assert_eq!(
        odd_host.profile(),
        Err("proxy host \"kubernetes.proxy.test/clusters\" is not a host name".into())
    );
    assert_eq!(
        quoted_token.profile(),
        Err("placeholder token \"token: \\\"injected\\\"\" is not a plain YAML scalar".into())
    );
}
