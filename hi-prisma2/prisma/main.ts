import Photon from "@generated/photon";

const photon = new Photon();

async function main() {
  await photon.connect();

  const newUser = await photon.users.create({
    data: {
      name: "Alice",
      email: `alice_${Math.random()}@prisma.io`,
      posts: {
        create: [{ title: "Hello Prisma 2" }]
      }
    }
  });
  console.log(newUser);

  const allUsers = await photon.users.findMany();
  console.log(allUsers);

  const newPost = await photon.posts.create({
    data: { title: "Hello Prisma 2" }
  });
  console.log(newPost);

  const allPosts = await photon.posts.findMany();
  console.log(allPosts);

  await photon.disconnect();
}

main();
