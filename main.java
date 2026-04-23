import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

/**
 * Main.java — Java port of the C++ VectorDB Engine
 *
 * Dependencies: Java 11+ (uses java.net.http.HttpClient, com.sun.net.httpserver)
 *
 * Compile:  javac Main.java
 * Run:      java Main
 *
 * Requires Ollama running locally for RAG endpoints:
 *   https://ollama.com
 *   ollama pull nomic-embed-text
 *   ollama pull llama3.2
 */
public class main {

    static final int DIMS = 16;

    // =====================================================================
    //  DATA TYPES
    // =====================================================================

    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    @FunctionalInterface
    interface DistFn {
        float apply(float[] a, float[] b);
    }

    // =====================================================================
    //  DISTANCE METRICS
    // =====================================================================

    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float) Math.sqrt(s);
    }

    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float)(Math.sqrt(na) * Math.sqrt(nb));
    }

    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m))    return main::cosine;
        if ("manhattan".equals(m)) return main::manhattan;
        return main::euclidean;
    }

    // =====================================================================
    //  BRUTE FORCE
    // =====================================================================

    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) { items.add(v); }

        List<float[]> knn(float[] q, int k, DistFn dist) {
            List<float[]> r = new ArrayList<>();
            for (VectorItem v : items) r.add(new float[]{dist.apply(q, v.emb), v.id});
            r.sort(Comparator.comparingDouble(x -> x[0]));
            return r.subList(0, Math.min(k, r.size()));
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =====================================================================
    //  KD-TREE
    // =====================================================================

    static class KDNode {
        VectorItem item;
        KDNode left, right;
        KDNode(VectorItem v) { this.item = v; }
    }

    static class KDTree {
        KDNode root;
        int dims;

        KDTree(int d) { this.dims = d; }

        KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left  = ins(n.left,  v, d + 1);
            else                             n.right = ins(n.right, v, d + 1);
            return n;
        }

        void insert(VectorItem v) { root = ins(root, v, 0); }

        void knnSearch(KDNode n, float[] q, int k, int d, DistFn dist,
                       PriorityQueue<float[]> heap) {
            if (n == null) return;
            float dn = dist.apply(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek()[0]) {
                heap.offer(new float[]{dn, n.item.id});
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer  = diff < 0 ? n.left  : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knnSearch(closer,  q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek()[0])
                knnSearch(farther, q, k, d + 1, dist, heap);
        }

        List<float[]> knn(float[] q, int k, DistFn dist) {
            // Max-heap by distance
            PriorityQueue<float[]> heap = new PriorityQueue<>(
                (a, b) -> Float.compare(b[0], a[0]));
            knnSearch(root, q, k, 0, dist, heap);
            List<float[]> r = new ArrayList<>(heap);
            r.sort(Comparator.comparingDouble(x -> x[0]));
            return r;
        }

        void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }

    // =====================================================================
    //  HNSW — Hierarchical Navigable Small World
    // =====================================================================

    static class HNSW {
        static class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;

            Node(VectorItem item, int maxLyr) {
                this.item   = item;
                this.maxLyr = maxLyr;
                this.nbrs   = new ArrayList<>();
                for (int i = 0; i <= maxLyr; i++) nbrs.add(new ArrayList<>());
            }
        }

        Map<Integer, Node> G = new HashMap<>();
        int    M, M0, efBuild;
        double mL;
        int    topLayer = -1;
        int    entryPt  = -1;
        Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            this.M       = m;
            this.M0      = 2 * m;
            this.efBuild = efBuild;
            this.mL      = 1.0 / Math.log(m);
        }

        int randLevel() {
            return (int) Math.floor(-Math.log(rng.nextFloat()) * mL);
        }

        List<float[]> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            // Min-heap for candidates
            PriorityQueue<float[]> cands = new PriorityQueue<>(Comparator.comparingDouble(x -> x[0]));
            // Max-heap for found
            PriorityQueue<float[]> found = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

            float d0 = dist.apply(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.offer(new float[]{d0, ep});
            found.offer(new float[]{d0, ep});

            while (!cands.isEmpty()) {
                float[] top = cands.poll();
                float cd = top[0]; int cid = (int) top[1];
                if (found.size() >= ef && cd > found.peek()[0]) break;
                Node cn = G.get(cid);
                if (cn == null || lyr >= cn.nbrs.size()) continue;
                for (int nid : cn.nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);
                    float nd = dist.apply(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek()[0]) {
                        cands.offer(new float[]{nd, nid});
                        found.offer(new float[]{nd, nid});
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<float[]> res = new ArrayList<>(found);
            res.sort(Comparator.comparingDouble(x -> x[0]));
            return res;
        }

        List<Integer> selectNbrs(List<float[]> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++)
                r.add((int) cands.get(i)[1]);
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id  = item.id;
            int lvl = randLevel();
            G.put(id, new Node(item, lvl));

            if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<float[]> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = (int) W.get(0)[1];
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<float[]> W = searchLayer(item.emb, ep, efBuild, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, new ArrayList<>(sel));

                for (int nid : sel) {
                    Node nn = G.get(nid);
                    if (nn == null) continue;
                    while (nn.nbrs.size() <= lc) nn.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nn.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<float[]> ds = new ArrayList<>();
                        for (int c : conn)
                            if (G.containsKey(c))
                                ds.add(new float[]{dist.apply(nn.item.emb, G.get(c).item.emb), c});
                        ds.sort(Comparator.comparingDouble(x -> x[0]));
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++)
                            conn.add((int) ds.get(i)[1]);
                    }
                }
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
            if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
        }

        List<float[]> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<float[]> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = (int) W.get(0)[1];
                }
            }
            List<float[]> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            return W.subList(0, Math.min(k, W.size()));
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values())
                for (List<Integer> layer : nd.nbrs)
                    layer.removeIf(n -> n == id);
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) if (nid != id) { entryPt = nid; break; }
            }
            G.remove(id);
        }

        // Graph info for visualization
        static class GraphInfo {
            int topLayer, nodeCount;
            int[] nodesPerLayer, edgesPerLayer;
            List<int[]> nodes = new ArrayList<>();  // {id, maxLyr}
            List<String> nodesMeta = new ArrayList<>();
            List<String> nodesCat  = new ArrayList<>();
            List<int[]> edges = new ArrayList<>();  // {src, dst, lyr}
        }

        GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer  = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            gi.nodesPerLayer = new int[maxL];
            gi.edgesPerLayer = new int[maxL];
            for (Map.Entry<Integer, Node> e : G.entrySet()) {
                int id = e.getKey(); Node nd = e.getValue();
                gi.nodes.add(new int[]{id, nd.maxLyr});
                gi.nodesMeta.add(nd.item.metadata);
                gi.nodesCat.add(nd.item.category);
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer[lc]++;
                    if (lc < nd.nbrs.size())
                        for (int nid : nd.nbrs.get(lc))
                            if (id < nid) {
                                gi.edgesPerLayer[lc]++;
                                gi.edges.add(new int[]{id, nid, lc});
                            }
                }
            }
            return gi;
        }

        int size() { return G.size(); }
    }

    // =====================================================================
    //  VECTOR DATABASE  (demo 16D index)
    // =====================================================================

    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf   = new BruteForce();
        KDTree     kdt;
        HNSW       hnsw = new HNSW(16, 200);
        Object     mu   = new Object();
        AtomicInteger nextId = new AtomicInteger(1);
        final int dims;

        VectorDB(int d) { this.dims = d; kdt = new KDTree(d); }

        int insert(String meta, String cat, float[] emb, DistFn dist) {
            synchronized (mu) {
                VectorItem v = new VectorItem(nextId.getAndIncrement(), meta, cat, emb);
                store.put(v.id, v);
                bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
                return v.id;
            }
        }

        boolean remove(int id) {
            synchronized (mu) {
                if (!store.containsKey(id)) return false;
                store.remove(id); bf.remove(id); hnsw.remove(id);
                kdt.rebuild(new ArrayList<>(store.values()));
                return true;
            }
        }

        static class Hit {
            int id; String meta, cat; float[] emb; float dist;
            Hit(int id, String meta, String cat, float[] emb, float dist) {
                this.id=id; this.meta=meta; this.cat=cat; this.emb=emb; this.dist=dist;
            }
        }

        static class SearchOut {
            List<Hit> hits; long us; String algo, metric;
            SearchOut(List<Hit> hits, long us, String algo, String metric) {
                this.hits=hits; this.us=us; this.algo=algo; this.metric=metric;
            }
        }

        SearchOut search(float[] q, int k, String metric, String algo) {
            synchronized (mu) {
                DistFn dfn = getDistFn(metric);
                long t0 = System.nanoTime();
                List<float[]> raw;
                if      ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
                else if ("kdtree".equals(algo))     raw = kdt.knn(q, k, dfn);
                else                                raw = hnsw.knn(q, k, 50, dfn);
                long us = (System.nanoTime() - t0) / 1000;
                List<Hit> hits = new ArrayList<>();
                for (float[] r : raw) {
                    int id = (int) r[1];
                    if (store.containsKey(id)) {
                        VectorItem v = store.get(id);
                        hits.add(new Hit(id, v.metadata, v.category, v.emb, r[0]));
                    }
                }
                return new SearchOut(hits, us, algo, metric);
            }
        }

        static class BenchOut { long bfUs, kdUs, hnswUs; int n; }

        BenchOut benchmark(float[] q, int k, String metric) {
            synchronized (mu) {
                DistFn dfn = getDistFn(metric);
                BenchOut b = new BenchOut();
                b.n = store.size();
                long t; final DistFn d = dfn;
                t = System.nanoTime(); bf.knn(q, k, d);   b.bfUs   = (System.nanoTime()-t)/1000;
                t = System.nanoTime(); kdt.knn(q, k, d);  b.kdUs   = (System.nanoTime()-t)/1000;
                t = System.nanoTime(); hnsw.knn(q,k,50,d);b.hnswUs = (System.nanoTime()-t)/1000;
                return b;
            }
        }

        List<VectorItem> all() {
            synchronized (mu) { return new ArrayList<>(store.values()); }
        }

        HNSW.GraphInfo hnswInfo() {
            synchronized (mu) { return hnsw.getInfo(); }
        }

        int size() { synchronized (mu) { return store.size(); } }
    }

    // =====================================================================
    //  TEXT CHUNKER
    // =====================================================================

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) return new ArrayList<>();
        if (words.length <= chunkWords) return Collections.singletonList(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) { if (j > i) sb.append(' '); sb.append(words[j]); }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    // =====================================================================
    //  OLLAMA CLIENT
    // =====================================================================

    static class OllamaClient {
        String host;
        int    port;
        String embedModel = "nomic-embed-text";
        String genModel   = "llama3.2";
        HttpClient http;

        OllamaClient(String host, int port) {
            this.host = host; this.port = port;
            this.http = HttpClient.newBuilder()
                .connectTimeout(java.time.Duration.ofSeconds(3)).build();
        }

        String esc(String s) {
            StringBuilder o = new StringBuilder();
            for (char c : s.toCharArray()) {
                if      (c == '"')  o.append("\\\"");
                else if (c == '\\') o.append("\\\\");
                else if (c == '\n') o.append("\\n");
                else if (c == '\r') o.append("\\r");
                else if (c == '\t') o.append("\\t");
                else                o.append(c);
            }
            return o.toString();
        }

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/tags"))
                    .timeout(java.time.Duration.ofSeconds(2))
                    .GET().build();
                HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch (Exception e) { return false; }
        }

        float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                    .timeout(java.time.Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .header("Content-Type", "application/json")
                    .build();
                HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return new float[0];
                return parseEmbedding(res.body());
            } catch (Exception e) { return new float[0]; }
        }

        float[] parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p < 0) return new float[0];
            p = body.indexOf('[', p);
            if (p < 0) return new float[0];
            int e = p + 1, depth = 1;
            while (e < body.length() && depth > 0) {
                if (body.charAt(e) == '[') depth++;
                else if (body.charAt(e) == ']') depth--;
                e++;
            }
            return parseVec(body.substring(p + 1, e - 1));
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\","
                            + "\"prompt\":\"" + esc(prompt) + "\","
                            + "\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                    .timeout(java.time.Duration.ofSeconds(180))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .header("Content-Type", "application/json")
                    .build();
                HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return extractStr(res.body(), "response");
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =====================================================================
    //  DOCUMENT DATABASE
    // =====================================================================

    static class DocItem {
        int id; String title, text; float[] emb;
        DocItem(int id, String title, String text, float[] emb) {
            this.id=id; this.title=title; this.text=text; this.emb=emb;
        }
    }

    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW       hnsw = new HNSW(16, 200);
        BruteForce bf   = new BruteForce();
        Object     mu   = new Object();
        AtomicInteger nextId = new AtomicInteger(1);
        volatile int dims = 0;

        int insert(String title, String text, float[] emb) {
            synchronized (mu) {
                if (dims == 0) dims = emb.length;
                DocItem item = new DocItem(nextId.getAndIncrement(), title, text, emb);
                store.put(item.id, item);
                VectorItem vi = new VectorItem(item.id, title, "doc", emb);
                hnsw.insert(vi, main::cosine);
                bf.insert(vi);
                return item.id;
            }
        }

        List<float[]> search(float[] q, int k, float maxDist) {
            synchronized (mu) {
                if (store.isEmpty()) return new ArrayList<>();
                List<float[]> raw = (store.size() < 10)
                    ? bf.knn(q, k, main::cosine)
                    : hnsw.knn(q, k, 50, main::cosine);
                return raw.stream().filter(r -> r[0] <= maxDist).collect(Collectors.toList());
            }
        }

        boolean remove(int id) {
            synchronized (mu) {
                if (!store.containsKey(id)) return false;
                store.remove(id); hnsw.remove(id); bf.remove(id);
                return true;
            }
        }

        List<DocItem> all() { synchronized (mu) { return new ArrayList<>(store.values()); } }
        int size()          { synchronized (mu) { return store.size(); } }
        int getDims()       { return dims; }
    }

    // =====================================================================
    //  JSON HELPERS
    // =====================================================================

    static String jS(String s) {
        if (s == null) return "\"\"";
        StringBuilder o = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if      (c == '"')  o.append("\\\"");
            else if (c == '\\') o.append("\\\\");
            else if (c == '\n') o.append("\\n");
            else if (c == '\r') o.append("\\r");
            else if (c == '\t') o.append("\\t");
            else                o.append(c);
        }
        return o.append('"').toString();
    }

    static String jVec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format("%.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    static float[] parseVec(String s) {
        List<Float> list = new ArrayList<>();
        for (String t : s.split(",")) {
            try { list.add(Float.parseFloat(t.trim())); } catch (NumberFormatException ignored) {}
        }
        float[] r = new float[list.size()];
        for (int i = 0; i < r.length; i++) r[i] = list.get(i);
        return r;
    }

    static String extractStr(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"':  result.append('"');  break;
                    case '\\': result.append('\\'); break;
                    case 'n':  result.append('\n'); break;
                    case 'r':  result.append('\r'); break;
                    case 't':  result.append('\t'); break;
                    default:   result.append(body.charAt(p)); break;
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    static int extractInt(String body, String key, int def) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        try {
            int end = p;
            while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) end++;
            return Integer.parseInt(body.substring(p, end));
        } catch (Exception e) { return def; }
    }

    static class ParsedBody {
        String meta, cat; float[] emb;
        boolean valid;
    }

    static ParsedBody parseBody(String b) {
        ParsedBody pb = new ParsedBody();
        pb.meta = extractStr(b, "metadata");
        pb.cat  = extractStr(b, "category");
        int p = b.indexOf("\"embedding\"");
        if (p >= 0) {
            p = b.indexOf('[', p);
            if (p >= 0) {
                int e = b.indexOf(']', p);
                if (e >= 0) pb.emb = parseVec(b.substring(p + 1, e));
            }
        }
        pb.valid = pb.meta != null && !pb.meta.isEmpty() && pb.emb != null && pb.emb.length > 0;
        return pb;
    }

    // =====================================================================
    //  HTTP HELPERS
    // =====================================================================

    static void cors(HttpExchange ex) throws IOException {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin",  "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    static void send(HttpExchange ex, int status, String body) throws IOException {
        ex.getResponseHeaders().set("Content-Type", "application/json");
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new HashMap<>();
        if (query == null || query.isEmpty()) return map;
        for (String p : query.split("&")) {
            String[] kv = p.split("=", 2);
            if (kv.length == 2) map.put(kv[0], URLDecoder.decode(kv[1], StandardCharsets.UTF_8));
        }
        return map;
    }

    // =====================================================================
    //  DEMO DATA LOADER
    // =====================================================================

    static void loadDemo(VectorDB db) {
        DistFn dist = getDistFn("cosine");
        // Dims 0-3: CS | 4-7: Math | 8-11: Food | 12-15: Sports
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }

    // =====================================================================
    //  MAIN — HTTP SERVER
    // =====================================================================

    public static void main(String[] args) throws Exception {
        VectorDB    db     = new VectorDB(DIMS);
        DocumentDB  docDB  = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) System.out.println("  embed model: " + ollama.embedModel
                                        + "  gen model: " + ollama.genModel);

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);

        // ── OPTIONS (CORS preflight) ──────────────────────────────────────
        server.createContext("/", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1); return;
            }
            // Serve index.html for GET /
            if ("GET".equals(ex.getRequestMethod()) && "/".equals(ex.getRequestURI().getPath())) {
                File f = new File("index.html");
                if (!f.exists()) { ex.sendResponseHeaders(404, -1); return; }
                byte[] bytes = java.nio.file.Files.readAllBytes(f.toPath());
                ex.getResponseHeaders().set("Content-Type", "text/html");
                ex.sendResponseHeaders(200, bytes.length);
                try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
            } else {
                send(ex, 404, "{\"error\":\"not found\"}");
            }
        });

        // ── GET /search ───────────────────────────────────────────────────
        server.createContext("/search", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            Map<String,String> params = parseQuery(ex.getRequestURI().getQuery());
            float[] q = parseVec(params.getOrDefault("v", ""));
            if (q.length != DIMS) { send(ex, 400, "{\"error\":\"need "+DIMS+"D vector\"}"); return; }
            int k = 5; try { k = Integer.parseInt(params.getOrDefault("k","5")); } catch (Exception ignored){}
            String metric = params.getOrDefault("metric", "cosine");
            String algo   = params.getOrDefault("algo",   "hnsw");

            VectorDB.SearchOut out = db.search(q, k, metric, algo);
            StringBuilder ss = new StringBuilder("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) ss.append(',');
                VectorDB.Hit h = out.hits.get(i);
                ss.append("{\"id\":").append(h.id)
                  .append(",\"metadata\":").append(jS(h.meta))
                  .append(",\"category\":").append(jS(h.cat))
                  .append(",\"distance\":").append(String.format("%.6f", h.dist))
                  .append(",\"embedding\":").append(jVec(h.emb)).append('}');
            }
            ss.append("],\"latencyUs\":").append(out.us)
              .append(",\"algo\":").append(jS(out.algo))
              .append(",\"metric\":").append(jS(out.metric)).append('}');
            send(ex, 200, ss.toString());
        });

        // ── POST /insert ──────────────────────────────────────────────────
        server.createContext("/insert", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            ParsedBody pb = parseBody(readBody(ex));
            if (!pb.valid || pb.emb.length != DIMS) { send(ex, 400, "{\"error\":\"invalid body\"}"); return; }
            int id = db.insert(pb.meta, pb.cat, pb.emb, getDistFn("cosine"));
            send(ex, 200, "{\"id\":" + id + "}");
        });

        // ── DELETE /delete/{id} ───────────────────────────────────────────
        server.createContext("/delete/", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            String path = ex.getRequestURI().getPath();
            try {
                int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
                boolean ok = db.remove(id);
                send(ex, 200, "{\"ok\":" + ok + "}");
            } catch (Exception e) { send(ex, 400, "{\"error\":\"invalid id\"}"); }
        });

        // ── GET /items ────────────────────────────────────────────────────
        server.createContext("/items", ex -> {
            cors(ex);
            List<VectorItem> items = db.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) ss.append(',');
                VectorItem v = items.get(i);
                ss.append("{\"id\":").append(v.id)
                  .append(",\"metadata\":").append(jS(v.metadata))
                  .append(",\"category\":").append(jS(v.category))
                  .append(",\"embedding\":").append(jVec(v.emb)).append('}');
            }
            ss.append(']');
            send(ex, 200, ss.toString());
        });

        // ── GET /benchmark ────────────────────────────────────────────────
        server.createContext("/benchmark", ex -> {
            cors(ex);
            Map<String,String> params = parseQuery(ex.getRequestURI().getQuery());
            float[] q = parseVec(params.getOrDefault("v", ""));
            if (q.length != DIMS) { send(ex, 400, "{\"error\":\"need "+DIMS+"D vector\"}"); return; }
            int k = 5; try { k = Integer.parseInt(params.getOrDefault("k","5")); } catch (Exception ignored){}
            String metric = params.getOrDefault("metric", "cosine");
            VectorDB.BenchOut b = db.benchmark(q, k, metric);
            send(ex, 200, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                        + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}");
        });

        // ── GET /hnsw-info ────────────────────────────────────────────────
        server.createContext("/hnsw-info", ex -> {
            cors(ex);
            HNSW.GraphInfo gi = db.hnswInfo();
            StringBuilder ss = new StringBuilder();
            ss.append("{\"topLayer\":").append(gi.topLayer)
              .append(",\"nodeCount\":").append(gi.nodeCount)
              .append(",\"nodesPerLayer\":[");
            for (int i = 0; i < gi.nodesPerLayer.length; i++) {
                if (i > 0) ss.append(','); ss.append(gi.nodesPerLayer[i]);
            }
            ss.append("],\"edgesPerLayer\":[");
            for (int i = 0; i < gi.edgesPerLayer.length; i++) {
                if (i > 0) ss.append(','); ss.append(gi.edgesPerLayer[i]);
            }
            ss.append("],\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) ss.append(',');
                int[] n = gi.nodes.get(i);
                ss.append("{\"id\":").append(n[0])
                  .append(",\"metadata\":").append(jS(gi.nodesMeta.get(i)))
                  .append(",\"category\":").append(jS(gi.nodesCat.get(i)))
                  .append(",\"maxLyr\":").append(n[1]).append('}');
            }
            ss.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) ss.append(',');
                int[] e = gi.edges.get(i);
                ss.append("{\"src\":").append(e[0]).append(",\"dst\":").append(e[1])
                  .append(",\"lyr\":").append(e[2]).append('}');
            }
            ss.append("]}");
            send(ex, 200, ss.toString());
        });

        // ── POST /doc/insert ──────────────────────────────────────────────
        server.createContext("/doc/insert", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            String body  = readBody(ex);
            String title = extractStr(body, "title");
            String text  = extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) { send(ex, 400, "{\"error\":\"need title and text\"}"); return; }

            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids   = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    send(ex, 503, "{\"error\":\"Ollama unavailable. Install from https://ollama.com "
                                + "then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                    ? title + " [" + (i+1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }
            StringBuilder ss = new StringBuilder("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) { if (i > 0) ss.append(','); ss.append(ids.get(i)); }
            ss.append("],\"chunks\":").append(chunks.size()).append(",\"dims\":").append(docDB.getDims()).append('}');
            send(ex, 200, ss.toString());
        });

        // ── DELETE /doc/delete/{id} ───────────────────────────────────────
        server.createContext("/doc/delete/", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            String path = ex.getRequestURI().getPath();
            try {
                int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
                boolean ok = docDB.remove(id);
                send(ex, 200, "{\"ok\":" + ok + "}");
            } catch (Exception e) { send(ex, 400, "{\"error\":\"invalid id\"}"); }
        });

        // ── GET /doc/list ─────────────────────────────────────────────────
        server.createContext("/doc/list", ex -> {
            cors(ex);
            List<DocItem> docs = docDB.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) ss.append(',');
                DocItem d = docs.get(i);
                String preview = d.text.length() > 120 ? d.text.substring(0,120) + "…" : d.text;
                int words = d.text.split("\\s+").length;
                ss.append("{\"id\":").append(d.id)
                  .append(",\"title\":").append(jS(d.title))
                  .append(",\"preview\":").append(jS(preview))
                  .append(",\"words\":").append(words).append('}');
            }
            ss.append(']');
            send(ex, 200, ss.toString());
        });

        // ── POST /doc/search ──────────────────────────────────────────────
        server.createContext("/doc/search", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            String body     = readBody(ex);
            String question = extractStr(body, "question");
            int    k        = extractInt(body, "k", 3);
            if (question.isEmpty()) { send(ex, 400, "{\"error\":\"need question\"}"); return; }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) { send(ex, 503, "{\"error\":\"Ollama unavailable\"}"); return; }

            List<float[]> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ss = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(',');
                int id = (int) hits.get(i)[1];
                float dist = hits.get(i)[0];
                DocItem d = null;
                for (DocItem di : docDB.all()) if (di.id == id) { d = di; break; }
                if (d == null) continue;
                ss.append("{\"id\":").append(id)
                  .append(",\"title\":").append(jS(d.title))
                  .append(",\"distance\":").append(String.format("%.4f", dist)).append('}');
            }
            ss.append("]}");
            send(ex, 200, ss.toString());
        });

        // ── POST /doc/ask ─────────────────────────────────────────────────
        server.createContext("/doc/ask", ex -> {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(204,-1); return; }
            String body     = readBody(ex);
            String question = extractStr(body, "question");
            int    k        = extractInt(body, "k", 3);
            if (question.isEmpty()) { send(ex, 400, "{\"error\":\"need question\"}"); return; }

            // Step 1: embed
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) { send(ex, 503, "{\"error\":\"Ollama unavailable\"}"); return; }

            // Step 2: retrieve
            List<float[]> rawHits = docDB.search(qEmb, k, 0.7f);
            List<DocItem> hitDocs = new ArrayList<>();
            List<Float>   hitDist = new ArrayList<>();
            for (float[] r : rawHits) {
                int id = (int) r[1];
                for (DocItem di : docDB.all()) if (di.id == id) { hitDocs.add(di); hitDist.add(r[0]); break; }
            }

            // Step 3: build prompt
            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hitDocs.size(); i++)
                ctx.append('[').append(i+1).append("] ").append(hitDocs.get(i).title)
                   .append(":\n").append(hitDocs.get(i).text).append("\n\n");

            String prompt = "You are a helpful assistant. Answer the user's question directly. "
                + "Use the provided context if it contains relevant information. "
                + "If it doesn't, just use your own general knowledge. "
                + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
                + "'the context doesn't mention'. Just answer the question naturally.\n\n"
                + "Context:\n" + ctx
                + "Question: " + question + "\n\nAnswer:";

            // Step 4: generate
            String answer = ollama.generate(prompt);

            // Step 5: return
            StringBuilder ss = new StringBuilder("{\"answer\":").append(jS(answer))
                .append(",\"model\":").append(jS(ollama.genModel))
                .append(",\"contexts\":[");
            for (int i = 0; i < hitDocs.size(); i++) {
                if (i > 0) ss.append(',');
                DocItem d = hitDocs.get(i);
                ss.append("{\"id\":").append(d.id)
                  .append(",\"title\":").append(jS(d.title))
                  .append(",\"text\":").append(jS(d.text))
                  .append(",\"distance\":").append(String.format("%.4f", hitDist.get(i))).append('}');
            }
            ss.append("],\"docCount\":").append(docDB.size()).append('}');
            send(ex, 200, ss.toString());
        });

        // ── GET /status ───────────────────────────────────────────────────
        server.createContext("/status", ex -> {
            cors(ex);
            boolean up = ollama.isAvailable();
            send(ex, 200, "{\"ollamaAvailable\":" + up
                + ",\"embedModel\":"  + jS(ollama.embedModel)
                + ",\"genModel\":"    + jS(ollama.genModel)
                + ",\"docCount\":"    + docDB.size()
                + ",\"docDims\":"     + docDB.getDims()
                + ",\"demoDims\":"    + DIMS
                + ",\"demoCount\":"   + db.size() + "}");
        });

        // ── GET /stats ────────────────────────────────────────────────────
        server.createContext("/stats", ex -> {
            cors(ex);
            send(ex, 200, "{\"count\":" + db.size()
                + ",\"dims\":"       + DIMS
                + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
        });

        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
        System.out.println("Server started on http://localhost:8080");
    }
}