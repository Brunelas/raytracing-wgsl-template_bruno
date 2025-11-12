// =========================
// raytracer.wgsl (Bruno) - versão enxuta + obj_id/kind + cores Fuzz/Spec
// =========================

const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_2_PI = 1.5707964f;

@group(0) @binding(0)
var<storage, read_write> fb : array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)
var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
var<storage, read_write> spheresb : array<sphere>;

@group(2) @binding(1)
var<storage, read_write> quadsb : array<quad>;

@group(2) @binding(2)
var<storage, read_write> boxesb : array<box>;

@group(2) @binding(3)
var<storage, read_write> trianglesb : array<triangle>;

@group(2) @binding(4)
var<storage, read_write> meshb : array<mesh>;

// ---------- Tipos ----------
struct ray {
  origin : vec3f,
  direction : vec3f,
};

struct sphere {
  transform : vec4f,   // xyz = centro, w = raio
  color : vec4f,
  material : vec4f,    // x: smooth (0..1 ou <0 p/vidro), y: reservado, z: specProb(0..1) ou IOR(>1.01), w: emissivo
};

struct quad {
  Q : vec4f, u : vec4f, v : vec4f,
  color : vec4f,
  material : vec4f,
};

struct box {
  center : vec4f, radius : vec4f, rotation: vec4f,
  color : vec4f,
  material : vec4f,
};

struct triangle {
  v0 : vec4f,
  v1 : vec4f,
  v2 : vec4f,
};

struct mesh {
  transform : vec4f,  // C (xyz)
  scale : vec4f,      // escala (xyz)
  rotation : vec4f,   // quat (x,y,z,w)
  color : vec4f,
  material : vec4f,
  min : vec4f, max : vec4f, // AABB local
  show_bb : f32,
  start : f32, end : f32,   // faixa de triângulos
};

struct material_behaviour {
  scatter  : bool,
  direction: vec3f,
};

struct camera {
  origin : vec3f,
  lower_left_corner : vec3f,
  horizontal : vec3f,
  vertical : vec3f,
  u : vec3f, v : vec3f, w : vec3f,
  lens_radius : f32,
};

struct hit_record {
  t : f32,
  p : vec3f,
  normal : vec3f,
  object_color : vec4f,
  object_material : vec4f,
  frontface : bool,
  hit_anything : bool,
  // NOVO:
  obj_id   : i32,   // índice do objeto dentro do tipo
  obj_kind : i32,   // 0=sphere, 1=quad, 2=box, 3=triangle
};

// ---------- Utilidades ----------
fn ray_at(r: ray, t: f32) -> vec3f {
  return r.origin + t * r.direction;
}

fn shifted_origin(p: vec3f, n: vec3f, out_dir: vec3f) -> vec3f {
  var bias = RAY_TMIN * 20.0;
  if (dot(out_dir, n) < 0.0) { bias = -bias; }
  return p + n * bias;
}

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray {
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(
    cam.origin + offset,
    normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset)
  );
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera {
  var camera = camera();
  camera.lens_radius = aperture / 2.0;

  let theta = degrees_to_radians(vfov);
  let h = tan(theta / 2.0);
  let w = aspect_ratio * h;

  camera.origin = lookfrom;
  camera.w = normalize(lookfrom - lookat);
  camera.u = normalize(cross(vup, camera.w));
  camera.v = cross(camera.u, camera.w);

  camera.lower_left_corner = camera.origin - w * focus_dist * camera.u - h * focus_dist * camera.v - focus_dist * camera.w;
  camera.horizontal = 2.0 * w * focus_dist * camera.u;
  camera.vertical   = 2.0 * h * focus_dist * camera.v;
  return camera;
}

fn envoriment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f {
  let unit_direction = normalize(direction);
  let t = 0.5 * (unit_direction.y + 1.0);
  var col = (1.0 - t) * color1 + t * color2;

  let sun_direction = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  let sun_color = int_to_rgb(i32(uniforms[17]));
  let sun_intensity = uniforms[16];
  let sun_size = uniforms[18];

  let sun = clamp(dot(sun_direction, unit_direction), 0.0, 1.0);
  col += sun_color * max(0.0, (pow(sun, sun_size) * sun_intensity));
  return col;
}

// ---------- BRDFs ----------
fn lambertian(normal : vec3f, _absorption: f32, random_sphere: vec3f, _rng_state: ptr<function, u32>) -> material_behaviour {
  var dir = normalize(normal + random_sphere);
  if (all(abs(dir) < vec3f(1e-6))) { dir = normal; }
  return material_behaviour(true, dir);
}

fn metal(normal : vec3f, direction: vec3f, fuzz: f32, random_sphere: vec3f) -> material_behaviour {
  let n = normalize(normal);
  let v = normalize(direction);
  var reflected = reflect(v, n);
  let f = clamp(fuzz, 0.0, 1.0);
  reflected = normalize(reflected + f * random_sphere);
  let scatter = dot(reflected, n) > 0.0;
  return material_behaviour(scatter, reflected);
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
  var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn dielectric( normal: vec3f, in_dir: vec3f, ior: f32, frontface: bool, random_sphere: vec3f, fuzz: f32, rng_state: ptr<function, u32>) -> material_behaviour {
  let n = normalize(normal);
  let v = normalize(in_dir);

  var eta = 1.0 / ior;
  if (!frontface) { eta = ior; }

  let cos_theta = min(dot(-v, n), 1.0);
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

  let F = schlick(cos_theta, ior);
  let cannot_refract = eta * sin_theta > 1.0; //Condicao TIR

  var dir = vec3f(0.0);
  if (cannot_refract || rng_next_float(rng_state) < F) {
    dir = normalize(reflect(v, n) + fuzz * random_sphere);  // usar fuzz=0.0 para vidro "limpo"
  } else {
    dir = refract(v, n, eta);
  }
  return material_behaviour(true, normalize(dir));
}

fn emmisive(_color: vec3f, _light: f32) -> material_behaviour {
  return material_behaviour(false, vec3f(0.0));
}

// ---------- Interseções ----------
fn check_ray_collision(r: ray, maxT: f32) -> hit_record {
  let spheresCount   = i32(uniforms[19]);
  let quadsCount     = i32(uniforms[20]);
  let boxesCount     = i32(uniforms[21]);
  let trianglesCount = i32(uniforms[22]);
  let meshCount      = i32(uniforms[27]);

  var closest = hit_record(
    maxT, vec3f(0.0), vec3f(0.0),
    vec4f(0.0), vec4f(0.0),
    false, false,
    -1, -1
  );

  // Esferas
  for (var i = 0; i < spheresCount; i = i + 1) {
    let S = spheresb[i];
    var tmp = hit_record(0.0, vec3f(0.0), vec3f(0.0),
                         vec4f(0.0), vec4f(0.0),
                         false, false,
                         -1, -1);
    hit_sphere(S.transform.xyz, S.transform.w, r, &tmp, min(maxT, closest.t));
    if (tmp.hit_anything && tmp.t < closest.t) {
      closest = hit_record(
        tmp.t, tmp.p, tmp.normal,
        S.color, S.material,
        tmp.frontface, true,
        i, 0    // obj_id, obj_kind=0 (sphere)
      );
    }
  }

  // Quads
  for (var qi = 0; qi < quadsCount; qi = qi + 1) {
    let Qd = quadsb[qi];
    var tmp = hit_record(0.0, vec3f(0.0), vec3f(0.0),
                         vec4f(0.0), vec4f(0.0),
                         false, false,
                         -1, -1);
    hit_quad(r, Qd.Q, Qd.u, Qd.v, &tmp, min(maxT, closest.t));
    if (tmp.hit_anything && tmp.t < closest.t) {
      let ff = dot(r.direction, tmp.normal) < 0.0;
      var n  = tmp.normal; if (!ff) { n = -n; }
      closest = hit_record(
        tmp.t, tmp.p, n,
        Qd.color, Qd.material,
        ff, true,
        qi, 1   // quad
      );
    }
  }

  // Caixas
  for (var bi = 0; bi < boxesCount; bi = bi + 1) {
    let B = boxesb[bi];
    var tmp = hit_record(0.0, vec3f(0.0), vec3f(0.0),
                         vec4f(0.0), vec4f(0.0),
                         false, false,
                         -1, -1);
    hit_box(r, B.center.xyz, B.radius.xyz, &tmp, min(maxT, closest.t));
    if (tmp.hit_anything && tmp.t < closest.t) {
      let ff = dot(r.direction, tmp.normal) < 0.0;
      var n  = tmp.normal; if (!ff) { n = -n; }
      closest = hit_record(
        tmp.t, tmp.p, n,
        B.color, B.material,
        ff, true,
        bi, 2   // box
      );
    }
  }

  // Triângulos e meshes (translate + scale)
  for (var ti = 0; ti < trianglesCount; ti = ti + 1) {
    let T = trianglesb[ti];

    var owner = -1;
    var C = vec3f(0.0);
    var S = vec3f(1.0);
    for (var mi = 0; mi < meshCount; mi = mi + 1) {
      let m = meshb[mi];
      let s = i32(m.start);
      let e = i32(m.end);
      if (ti >= s && ti < e) { owner = mi; C = m.transform.xyz; S = m.scale.xyz; break; }
    }

    var v0w = T.v0.xyz; var v1w = T.v1.xyz; var v2w = T.v2.xyz;
    if (owner >= 0) { v0w = C + S * v0w; v1w = C + S * v1w; v2w = C + S * v2w; }

    var tmp = hit_record(0.0, vec3f(0.0), vec3f(0.0),
                         vec4f(0.0), vec4f(0.0),
                         false, false,
                         -1, -1);
    hit_triangle(r, v0w, v1w, v2w, &tmp, min(maxT, closest.t));
    if (tmp.hit_anything && tmp.t < closest.t) {
      tmp.frontface = dot(r.direction, tmp.normal) < 0.0;

      var triColor    = vec4f(1.0, 1.0, 1.0, 1.0);
      var triMaterial = vec4f(0.0, 0.0, 0.0, 0.0);
      if (owner >= 0) {
        let mm = meshb[owner];
        triColor = mm.color; triMaterial = mm.material;
      }

      closest = hit_record(
        tmp.t, tmp.p, tmp.normal,
        triColor, triMaterial,
        tmp.frontface, true,
        ti, 3   // triangle
      );
    }
  }

  return closest;
}

// ---------- Integrador ----------
fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f {
  let maxbounces = i32(uniforms[2]);
  let bg1 = int_to_rgb(i32(uniforms[11]));
  let bg2 = int_to_rgb(i32(uniforms[12]));

  var throughput = vec3f(1.0);
  var light = vec3f(0.0);
  var ray_ = r;

  for (var j = 0; j < maxbounces; j = j + 1) {
    let rec = check_ray_collision(ray_, RAY_TMAX); //grava o hit_record para o raio atual ate a distancia maxima

    // Fundo (céu)
    if (!rec.hit_anything) {
      light += throughput * envoriment_color(ray_.direction, bg1, bg2); // aqui verifica se nao bateu em nada e quebra
      break;
    }

    // Emissivo
    let emissive = rec.object_material.w; 
    if (emissive > 0.0) {
      light += throughput * rec.object_color.xyz * emissive; // ve se é uma fonte de luz
      break;
    }

    // x = smooth (0..1 ou <0 p/vidro), z = specProb (0..1) ou IOR (>1.01)
    let smooth_val  = clamp(rec.object_material.x, 0.0, 1.0);
    let spec_or_ior = rec.object_material.z; //aqui define o material pode ser especular ou Indice de refracao

    let rand_sph = rng_next_vec3_in_unit_sphere(rng_state); //usado para gerar um numero aleatroia para o monte carlo

    // Vidro (usa dielectric)
    let is_glass = (rec.object_material.x < 0.0) || (spec_or_ior > 1.01);
    if (is_glass) {
      let ior = max(1.01, spec_or_ior);
      let mb = dielectric(rec.normal, ray_.direction, ior, rec.frontface, rand_sph, 0.0, rng_state); // fuzz=0.0
      throughput *= rec.object_color.xyz;

      let nd = normalize(mb.direction);
      // ray_ = ray(rec.p + nd * RAY_TMIN, nd);  //uso o shifted_origin aqui mas de uma forma sem a funcao
      let org = shifted_origin(rec.p, rec.normal, nd);
      ray_ = ray(org, nd);
      continue;
    }

    // Glossy + difuso (probabilidade em z)
    let spec_prob = clamp(spec_or_ior, 0.0, 1.0);
    let roughness = 1.0 - smooth_val;     // 0 = liso, 1 = áspero
    var  refl_fuzz = 0.35 * roughness;

    // -- cena Fuzz (se usar material.y como fuzz da cena)
    if (spec_prob > 0.99) {
      let fuzz_override = clamp(rec.object_material.y, 0.0, 1.0);
      refl_fuzz = fuzz_override;
    }

    // Amostragem: se cair em especular (metal), NÃO “tintar”; se cair em difuso, aplica albedo
    var out_dir = vec3f(0.0);
    if (rng_next_float(rng_state) < spec_prob) {
      let mb = metal(rec.normal, ray_.direction, refl_fuzz, rand_sph);
      out_dir = mb.direction;               // especular → sem albedo (cromado/branco)
    } else {
      let mb = lambertian(rec.normal, 0.0, rand_sph, rng_state);
      out_dir = mb.direction;
      // difuso → aqui sim aplica a “tinta” do objeto
      throughput *= rec.object_color.xyz;
    }

    // avança o raio
    let org = shifted_origin(rec.p, rec.normal, out_dir);
    ray_ = ray(org, normalize(out_dir));



    // Russian roulette simples
    if (j >= 5) {
      let maxc = max(max(throughput.x, throughput.y), throughput.z);
      if (maxc < 1e-3) { break; }
      let p = clamp(maxc, 0.05, 0.99);
      if (rng_next_float(rng_state) > p) { break; }
      throughput /= p;
    }
  }

  return light;
}

// ---------- Kernel ----------
@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u) {
  let rez   = uniforms[1];
  let frame = u32(uniforms[0]);
  let spp   = max(1, i32(uniforms[4]));
  let doAcc = uniforms[3] > 0.5;

  var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), frame);

  let lookfrom = vec3(uniforms[7],  uniforms[8],  uniforms[9]);
  let lookat   = vec3(uniforms[23], uniforms[24], uniforms[25]);
  let cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0),
                       uniforms[10], 1.0, uniforms[6], uniforms[5]);

  let frag = vec2f(f32(id.x), f32(id.y));
  let idx  = mapfb(id.xy, rez);

  var acc = vec3f(0.0);
  for (var s = 0; s < spp; s = s + 1) {
    let uv = (frag + sample_square(&rng_state)) / vec2(rez);
    let col = trace(get_ray(cam, uv, &rng_state), &rng_state); // linear aplica raio e comeca o loop de bounce
    acc += col;
  }
  var frameLinear = acc / f32(spp);

  // Anti-firefly simples em luminância
  let L = dot(frameLinear, vec3f(0.2126, 0.7152, 0.0722));
  let Lmax = 6.0;
  if (L > Lmax) {
    frameLinear *= Lmax / max(L, 1e-6);
  }

  if (doAcc) {
    let prev = rtfb[idx];
    let sum  = prev.xyz + frameLinear;
    let w    = prev.w + 1.0;
    rtfb[idx] = vec4(sum, w);
    let outLin = sum / max(w, 1.0);
    fb[idx] = vec4(linear_to_gamma(outLin), 1.0);
  } else {
    rtfb[idx] = vec4(frameLinear, 1.0);
    fb[idx]   = vec4(linear_to_gamma(frameLinear), 1.0);
  }
}