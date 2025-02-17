import * as THREE from 'https://unpkg.com/three@0.127.0/build/three.module.js';
import {OrbitControls} from 'https://unpkg.com/three@0.127.0/examples/jsm/controls/OrbitControls.js'
import {GLTFLoader} from 'https://unpkg.com/three@0.127.0/examples/jsm/loaders/GLTFLoader.js';
const canvas = document.querySelector('canvas.webgl')

const scene = new THREE.Scene();

const textureLoader = new THREE.TextureLoader()
const myTexture = textureLoader.load('coolTex.jpg')
const monkeyUrl = new URL('./test.glb', import.meta.url);

let mixer;
const assetLoader = new GLTFLoader();
assetLoader.load(monkeyUrl.href, function(gltf) {
    const model = gltf.scene;
    scene.add(model);
    mixer = new THREE.AnimationMixer(model);
    const clips = gltf.animations;
    clips.forEach(function(clip) {
        const action = mixer.clipAction(clip);
        action.play();
    });

}, undefined, function(error) {
    console.error(error);
});
const sizes = {
    width:window.innerWidth,
    height:window.innerHeight
}



window.addEventListener('resize',()=>{
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    camera.aspect = sizes.width/sizes.height
    camera.updateProjectionMatrix()

    renderer.setSize(sizes.width/2.,sizes.height/2.)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2))
    
})


const camera = new THREE.PerspectiveCamera(45,sizes.width/sizes.height,0.1,100)
camera.position.set(3, 2, -3);
// camera.position.z = 3
scene.add(camera);

// const light = new THREE.AmbientLight( 0xF0F040 ); // soft white light
// scene.add( light );
// var hemiLight = new THREE.HemisphereLight( 0xffffff, 0xffffff, 0.6 );
// hemiLight.color.setHSV( 0.6, 0.75, 0.5 );
// hemiLight.groundColor.setHSV( 0.095, 0.5, 0.5 );
// hemiLight.position.set( 0, 500, 0 );
// scene.add( hemiLight );

const spotLight = new THREE.SpotLight(0xFFFFFF);
spotLight.position.set(50, 100, -100)
scene.add(spotLight)


const controls = new OrbitControls(camera, canvas);

controls.enableZoom = true;
controls.enableDamping = true;
controls.object.position.set(camera.position.x, camera.position.y, camera.position.z);
controls.target = new THREE.Vector3(0, 0, 0);

const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
})
renderer.setSize(window.innerWidth/2.5, window.innerHeight/2.5);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.outputEncoding = THREE.sRGBEncoding;

const clock = new THREE.Clock();


controls.update();

const grid = new THREE.GridHelper(30, 30);
scene.add(grid);





function animate() {
    if(mixer)
        mixer.update(clock.getDelta());
    renderer.render(scene, camera);
}

renderer.setAnimationLoop(animate);

