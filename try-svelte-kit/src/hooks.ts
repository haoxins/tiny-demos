import cookie from 'cookie';
import { v4 as uuid } from '@lukeed/uuid';
import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ request, resolve }) => {
  const cookies = cookie.parse(request.headers.cookie || '');
  request.locals.userid = cookies.userid || uuid();

  const response = await resolve(request);

  if (!cookies.userid) {
    response.headers['set-cookie'] = cookie.serialize('userid', request.locals.userid, {
      path: '/',
      httpOnly: true
    });
  }

  return response;
};
